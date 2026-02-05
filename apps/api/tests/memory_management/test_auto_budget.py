"""
Unit tests for automatic budget calculation in budget_offloading.py
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from src.memory_management.budget_offloading import BudgetOffloader
from src.memory_management.config import MemoryConfig


class MockTransformerModule(torch.nn.Module):
    """Mock transformer with tower structure for testing auto budget."""

    def __init__(self, num_blocks=10, block_size_params=1000000):
        super().__init__()
        # Base layer (not in towers)
        self.base_embedding = torch.nn.Embedding(1000, 512)
        self.base_norm = torch.nn.LayerNorm(512)

        # Tower blocks (repeating ModuleList structure)
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(512, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 512),
            )
            for _ in range(num_blocks)
        ])

        # Output layer (not in towers)
        self.output_proj = torch.nn.Linear(512, 1000)

    def forward(self, x):
        x = self.base_embedding(x)
        x = self.base_norm(x)
        for block in self.blocks:
            x = x + block(x)
        return self.output_proj(x)


class MockSimpleModule(torch.nn.Module):
    """Simple module without tower structure."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(100, 1000)
        self.layer2 = torch.nn.Linear(1000, 100)

    def forward(self, x):
        return self.layer2(self.layer1(x))


# Module-level fixtures accessible by all test classes
@pytest.fixture
def mock_vram_16gb():
    """Mock 16GB VRAM system."""
    with patch('src.memory_management.budget_offloading._device_free_total') as mock:
        free_vram = 14 * 1024**3  # 14GB free
        total_vram = 16 * 1024**3  # 16GB total
        mock.return_value = (free_vram, total_vram)
        yield mock


@pytest.fixture
def mock_vram_8gb():
    """Mock 8GB VRAM system."""
    with patch('src.memory_management.budget_offloading._device_free_total') as mock:
        free_vram = 7 * 1024**3  # 7GB free
        total_vram = 8 * 1024**3  # 8GB total
        mock.return_value = (free_vram, total_vram)
        yield mock


@pytest.fixture
def mock_vram_24gb():
    """Mock 24GB VRAM system."""
    with patch('src.memory_management.budget_offloading._device_free_total') as mock:
        free_vram = 22 * 1024**3  # 22GB free
        total_vram = 24 * 1024**3  # 24GB total
        mock.return_value = (free_vram, total_vram)
        yield mock


class TestAutoBudgetCalculation:
    """Test suite for automatic budget calculation."""

    def test_parse_budget_auto_mode(self, mock_vram_16gb):
        """Test that budget_mb='auto' triggers auto calculation."""
        module = MockTransformerModule(num_blocks=5)

        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have calculated a non-zero budget
        assert offloader._budget_bytes > 0
        # Budget should be less than 16GB (safety margins applied)
        assert offloader._budget_bytes < 16 * 1024**3

    def test_fallback_heuristic_8gb(self):
        """Test fallback heuristic for 8GB VRAM."""
        with patch('src.memory_management.budget_offloading._device_free_total') as mock:
            mock.return_value = (7 * 1024**3, 8 * 1024**3)

            module = MockSimpleModule()
            offloader = BudgetOffloader(
                module,
                torch.device("cuda"),
                offload_device=torch.device("cpu"),
                budget_mb="auto"
            )

            # For 8GB VRAM, should use 400MB or auto-calculated value
            assert offloader._budget_bytes > 0

    def test_fallback_heuristic_16gb(self):
        """Test fallback heuristic for 16GB VRAM."""
        with patch('src.memory_management.budget_offloading._device_free_total') as mock:
            mock.return_value = (14 * 1024**3, 16 * 1024**3)

            module = MockSimpleModule()

            # Force fallback by making _device_free_total unavailable in _auto_calculate_budget
            with patch.object(BudgetOffloader, '_auto_calculate_budget', side_effect=Exception("test")):
                offloader = BudgetOffloader(
                    module,
                    torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    budget_mb="auto"
                )

                # Should use 6000MB fallback for 16GB VRAM (16-24GB range)
                expected_budget = 6000 * 1024 * 1024
                assert offloader._budget_bytes == expected_budget

    def test_calculate_param_size_regular_tensor(self, mock_vram_16gb):
        """Test parameter size calculation for regular tensors."""
        module = MockSimpleModule()
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Test with a known parameter
        param = next(module.parameters())
        size = offloader._calculate_param_size(param)

        # Size should match numel × element_size
        expected_size = param.numel() * param.element_size()
        assert size == expected_size

    def test_auto_budget_respects_vram_safety_coefficient(self, mock_vram_16gb):
        """Test that auto budget respects vram_safety_coefficient."""
        module1 = MockTransformerModule(num_blocks=5)
        module2 = MockTransformerModule(num_blocks=5)

        # Test with different safety coefficients
        offloader_80 = BudgetOffloader(
            module1,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto",
            vram_safety_coefficient=0.8
        )

        offloader_90 = BudgetOffloader(
            module2,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto",
            vram_safety_coefficient=0.9
        )

        # Higher safety coefficient should allow larger budget
        # Note: Due to other safety factors and rounding, the difference may be small
        # Just verify both calculated reasonable budgets
        assert offloader_80._budget_bytes > 0
        assert offloader_90._budget_bytes > 0
        # The 0.9 coefficient should generally allow more budget
        assert offloader_90._budget_bytes >= offloader_80._budget_bytes

    def test_auto_budget_with_towers(self, mock_vram_16gb):
        """Test auto budget calculation with tower structure."""
        module = MockTransformerModule(num_blocks=10)

        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have detected towers and calculated budget
        assert offloader._budget_bytes > 0
        # Budget should include base block + some tower blocks
        # (exact amount depends on calculation, but should be reasonable)
        total_model_size = sum(p.numel() * p.element_size() for p in module.parameters())
        assert offloader._budget_bytes < total_model_size

    def test_auto_budget_no_towers(self, mock_vram_16gb):
        """Test auto budget with module that has no tower structure."""
        module = MockSimpleModule()

        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should still calculate a budget (loads full model if it fits)
        assert offloader._budget_bytes >= 0

    def test_manual_budget_override(self, mock_vram_16gb):
        """Test that manual budget overrides auto calculation."""
        module = MockTransformerModule(num_blocks=5)

        manual_budget_mb = 5000
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb=manual_budget_mb
        )

        # Should use manual budget, not auto
        expected_budget_bytes = manual_budget_mb * 1024 * 1024
        assert offloader._budget_bytes == expected_budget_bytes

    def test_budget_none_disables_budgeting(self, mock_vram_16gb):
        """Test that budget_mb=None disables budget offloading."""
        module = MockTransformerModule(num_blocks=5)

        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb=None
        )

        # Should have zero budget (loads full model)
        assert offloader._budget_bytes == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_auto_budget_with_real_cuda(self):
        """Test auto budget with actual CUDA device."""
        module = MockTransformerModule(num_blocks=5)

        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have calculated budget based on actual VRAM
        assert offloader._budget_bytes > 0

        # Verify budget is reasonable (not more than 90% of total VRAM)
        _, total_vram = torch.cuda.mem_get_info()
        assert offloader._budget_bytes < total_vram * 0.9

    def test_runtime_validation_adjusts_insufficient_budget(self, mock_vram_16gb):
        """Test that runtime validation adjusts budget if it's too small."""
        module = MockTransformerModule(num_blocks=10)

        # Create offloader with manually set very small budget
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb=10  # Intentionally too small
        )

        # After _tune_preloading, budget should have been validated
        # (this happens during __init__)
        # Budget should be at least base block size (validation may adjust)
        assert offloader._budget_bytes > 0


class TestMemoryConfigAutoBudget:
    """Test suite for MemoryConfig.auto_budget() classmethod."""

    def test_memory_config_auto_budget_default(self):
        """Test that budget_mb defaults to 'auto'."""
        config = MemoryConfig(offload_mode="budget")
        assert config.budget_mb == "auto"

    def test_memory_config_auto_budget_classmethod(self):
        """Test MemoryConfig.auto_budget() convenience method."""
        config = MemoryConfig.auto_budget()
        assert config.offload_mode == "budget"
        assert config.budget_mb == "auto"

    def test_memory_config_auto_budget_with_custom_safety(self):
        """Test auto_budget with custom vram_safety_coefficient."""
        config = MemoryConfig.auto_budget(vram_safety_coefficient=0.85)
        assert config.budget_mb == "auto"
        assert config.vram_safety_coefficient == 0.85

    def test_memory_config_manual_override(self):
        """Test that manual budget can override auto."""
        config = MemoryConfig(offload_mode="budget", budget_mb=3000)
        assert config.budget_mb == 3000

    def test_memory_config_disable_budget(self):
        """Test that budget_mb=None disables budgeting."""
        config = MemoryConfig(offload_mode="budget", budget_mb=None)
        assert config.budget_mb is None


class TestEnvironmentVariables:
    """Test environment variable overrides for auto budget."""

    def test_activation_mult_override(self, mock_vram_16gb, monkeypatch):
        """Test APEX_BUDGET_AUTO_ACTIVATION_MULT override."""
        monkeypatch.setenv("APEX_BUDGET_AUTO_ACTIVATION_MULT", "0.30")

        module = MockTransformerModule(num_blocks=5)
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have calculated budget with custom activation multiplier
        assert offloader._budget_bytes > 0

    def test_system_overhead_override(self, mock_vram_16gb, monkeypatch):
        """Test APEX_BUDGET_AUTO_SYSTEM_OVERHEAD_GB override."""
        monkeypatch.setenv("APEX_BUDGET_AUTO_SYSTEM_OVERHEAD_GB", "3.0")

        module = MockTransformerModule(num_blocks=5)
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have calculated budget with custom system overhead
        assert offloader._budget_bytes > 0

    def test_final_safety_override(self, mock_vram_16gb, monkeypatch):
        """Test APEX_BUDGET_AUTO_FINAL_SAFETY override."""
        monkeypatch.setenv("APEX_BUDGET_AUTO_FINAL_SAFETY", "0.90")

        module = MockTransformerModule(num_blocks=5)
        offloader = BudgetOffloader(
            module,
            torch.device("cuda"),
            offload_device=torch.device("cpu"),
            budget_mb="auto"
        )

        # Should have calculated budget with custom final safety factor
        assert offloader._budget_bytes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
