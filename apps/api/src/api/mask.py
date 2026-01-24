"""
API endpoints for mask generation using SAM2
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import numpy as np
from loguru import logger
import cv2
from datetime import datetime
import os

from src.mask.mask import get_sam2_predictor, ModelType, debug_save_rectangles

router = APIRouter(prefix="/mask", tags=["mask"])

# In-memory registry for cooperative cancellation of tracking streams
# Keyed by mask id provided by the client
CANCEL_TRACKING: set[str] = set()


class MaskRequest(BaseModel):
    """Request model for mask generation."""

    id: Optional[str] = Field(None, description="ID of the mask")
    input_path: str = Field(..., description="Path to input image or video file")
    frame_number: Optional[int] = Field(
        None, description="Frame number for video input (0-based, required for videos)"
    )
    tool: Literal["brush", "touch", "lasso", "shape"] = Field(
        ..., description="Masking tool type"
    )

    # Tool-specific data
    points: Optional[List[Dict[str, float]]] = Field(
        None, description="List of points with x, y coordinates (for brush/touch/lasso)"
    )
    point_labels: Optional[List[int]] = Field(
        None,
        description="Labels for points: 1=positive, 0=negative (optional, defaults to all positive)",
    )
    box: Optional[Dict[str, float]] = Field(
        None,
        description="Bounding box with x1, y1, x2, y2 (for shape tool or as additional constraint)",
    )

    # SAM2 parameters
    multimask_output: bool = Field(
        True,
        description="Whether to generate multiple mask proposals and return the best one",
    )
    simplify_tolerance: float = Field(
        1.0, description="Contour simplification tolerance (Douglas-Peucker epsilon)"
    )
    # Debug parameter
    debug: bool = Field(
        False,
        description="Enable debug visualization - saves input points/bbox and output contours as images",
    )
    shape_type: Optional[
        Literal["rectangle", "ellipse", "polygon", "triangle", "star"]
    ] = Field(
        None,
        description="Optional shape type for shape bounds from the provided box/points",
    )


class MaskTrackingRequest(BaseModel):
    """Request model for mask tracking."""

    id: str = Field(..., description="ID of the mask")
    input_path: str = Field(..., description="Path to input image or video file")
    frame_start: int = Field(
        ..., description="Start frame number for video input (0-based)"
    )
    frame_end: int = Field(
        ..., description="End frame number for video input (0-based)"
    )
    max_frames: Optional[int] = Field(
        None,
        description="Optional maximum number of frames to track from the anchor frame",
    )
    anchor_frame: Optional[int] = Field(
        None, description="Anchor frame containing the initial mask"
    )
    direction: Optional[Literal["forward", "backward", "both"]] = Field(
        None,
        description="Tracking direction: forward, backward, or both. Defaults inferred from frame range",
    )
    shape_type: Optional[
        Literal["rectangle", "ellipse", "polygon", "triangle", "star"]
    ] = Field(None, description="Optional shape type for shape bounds normalization")
    debug: bool = Field(
        False,
        description="Enable debug visualization - saves input points/bbox and output contours as images",
    )


class MaskResponse(BaseModel):
    """Response model for mask generation."""

    status: str = Field(..., description="Status: success or error")
    contours: Optional[List[List[float]]] = Field(
        None,
        description="List of contour polygons, each as flat list [x1, y1, x2, y2, ...]",
    )
    message: Optional[str] = Field(None, description="Success or error message")
    input_path: Optional[str] = Field(None, description="Echo of input path")
    frame_number: Optional[int] = Field(
        None, description="Echo of frame number (for videos)"
    )
    tool: Optional[str] = Field(None, description="Echo of tool used")
    direction: Optional[str] = Field(
        None, description="Direction of tracking: forward, backward, or both"
    )


def validate_input_file(input_path: str) -> tuple[Path, bool]:
    """
    Validate input file exists and determine if it's a video.

    Returns:
        Tuple of (Path object, is_video boolean)
    """
    path = Path(input_path)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Input file not found: {input_path}"
        )

    # Check if it's a video based on extension
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
    is_video = path.suffix.lower() in video_extensions

    return path, is_video


def prepare_mask_inputs(request: MaskRequest, image_shape: tuple) -> dict:
    """
    Prepare SAM2 inputs based on the masking tool and request parameters.

    Args:
        request: The mask request
        image_shape: Shape of the input image (H, W, C)

    Returns:
        Dictionary with point_coords, point_labels, and box arrays
    """
    point_coords = None
    point_labels = None
    box = None

    if request.tool in ["brush", "touch", "lasso"]:
        # Convert points to numpy array
        if request.points:
            coords = np.array(
                [[p["x"], p["y"]] for p in request.points], dtype=np.float32
            )

            # Handle point labels
            if request.point_labels:
                labels = np.array(request.point_labels, dtype=np.int32)
            else:
                # Default: all points are positive (include)
                labels = np.ones(len(coords), dtype=np.int32)

            point_coords = coords
            point_labels = labels

            logger.debug(
                f"Tool {request.tool}: {len(coords)} points with labels {labels}"
            )

    if request.tool == "shape" or request.box:
        # Convert box to numpy array
        if request.box:
            box = np.array(
                [
                    request.box["x1"],
                    request.box["y1"],
                    request.box["x2"],
                    request.box["y2"],
                ],
                dtype=np.float32,
            )

            logger.debug(f"Tool {request.tool}: box {box}")

    if request.tool == "lasso" and request.points:
        # For lasso, we can also compute a bounding box from the points
        # This helps SAM2 focus on the region of interest
        coords = np.array([[p["x"], p["y"]] for p in request.points])
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

        logger.debug(f"Lasso tool: computed box from {len(coords)} points: {box}")

    return {"point_coords": point_coords, "point_labels": point_labels, "box": box}


def save_debug_visualizations(
    input_path: Path,
    frame_number: Optional[int],
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    box: Optional[np.ndarray],
    contours: Optional[List[List[float]]],
    mask: Optional[np.ndarray],
    seed_mask: Optional[np.ndarray] = None,
):
    """
    Save debug visualizations to debug/ folder with a subfolder per API call.

    Args:
        input_path: Path to input image/video
        frame_number: Frame number for videos (None for images)
        point_coords: Array of point coordinates (N, 2)
        point_labels: Array of point labels (N,) where 1=positive, 0=negative
        box: Bounding box array [x1, y1, x2, y2]
        contours: List of contour polygons
        mask: Binary mask from SAM2 (H, W)
    """
    try:
        # Create debug folder with timestamp subfolder for this API call
        debug_base = Path("debug")
        debug_base.mkdir(exist_ok=True)

        # Generate timestamp for unique subfolder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = debug_base / timestamp
        debug_dir.mkdir(exist_ok=True)

        # Load the image/frame
        if frame_number is not None:
            # Load specific frame from video
            logger.info(f"Loading frame {frame_number} from video {input_path}")
            cap = cv2.VideoCapture(str(input_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, image = cap.read()
            cap.release()
            if not ret:
                logger.error(f"Failed to read frame {frame_number} from video")
                return
        else:
            # Load image
            logger.info(f"Loading image from {input_path}")
            image = cv2.imread(str(input_path))
            if image is None:
                logger.error(f"Failed to load image from {input_path}")
                return

        # === 1. Draw input visualization (points + bbox) ===
        input_vis = image.copy()

        # Draw bounding box if present
        if box is not None:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(input_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                input_vis,
                "BOX",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        # Draw points with labels
        if point_coords is not None and point_labels is not None:
            for (x, y), label in zip(point_coords, point_labels):
                x, y = int(x), int(y)

                # Color and symbol based on label
                if label == 1:  # Positive point
                    color = (0, 255, 0)  # Green
                    symbol = "+"
                else:  # Negative point (0)
                    color = (0, 0, 255)  # Red
                    symbol = "-"

                # Draw circle
                cv2.circle(input_vis, (x, y), 8, color, -1)
                cv2.circle(input_vis, (x, y), 10, (255, 255, 255), 2)

                # Draw symbol
                font_scale = 1.0
                thickness = 3
                text_size = cv2.getTextSize(
                    symbol, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )[0]
                text_x = x - text_size[0] // 2
                text_y = y + text_size[1] // 2

                # White background for better visibility
                cv2.putText(
                    input_vis,
                    symbol,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    thickness + 2,
                )
                cv2.putText(
                    input_vis,
                    symbol,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                )

        # Save input visualization
        input_path_debug = debug_dir / "input.png"
        cv2.imwrite(str(input_path_debug), input_vis)
        logger.info(f"Saved input visualization to {input_path_debug}")

        # === 2. Draw output visualization (contours) ===
        if contours:
            output_vis = image.copy()

            # Create an overlay for translucent effect
            overlay = output_vis.copy()

            # Draw each contour
            for contour in contours:
                # Convert flat list to points array
                points = np.array(contour).reshape(-1, 2).astype(np.int32)

                # Fill the contour with blue color
                cv2.fillPoly(overlay, [points], (255, 0, 0))  # BGR: Blue

                # Draw contour outline
                cv2.polylines(output_vis, [points], True, (255, 0, 0), 2)

            # Blend the overlay with the original for translucency
            alpha = 0.4  # Transparency factor
            output_vis = cv2.addWeighted(overlay, alpha, output_vis, 1 - alpha, 0)

            # Save output visualization
            output_path_debug = debug_dir / "output.png"
            cv2.imwrite(str(output_path_debug), output_vis)
            logger.info(f"Saved output visualization to {output_path_debug}")

        # === 3. Draw seed mask visualization if provided ===
        if seed_mask is not None:
            try:
                seed_vis = image.copy()
                overlay = seed_vis.copy()
                # Assume seed_mask is binary with 1 where shape exists
                seed_bool = seed_mask > 0
                overlay[seed_bool] = (
                    0.6 * overlay[seed_bool]
                    + 0.4 * np.array([0, 255, 0], dtype=np.float32)
                ).astype(np.uint8)
                seed_vis = overlay
                seed_path_debug = debug_dir / "seed.png"
                cv2.imwrite(str(seed_path_debug), seed_vis)
                logger.info(f"Saved seed mask visualization to {seed_path_debug}")
            except Exception as e:
                logger.warning(f"Failed to save seed mask visualization: {e}")

    except Exception as e:
        logger.error(f"Error saving debug visualizations: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


@router.post("/create", response_model=MaskResponse)
async def create_mask(request: MaskRequest):
    """
    Generate a mask using SAM2 based on the provided inputs.

    Supports:
    - Image or video inputs (for videos, specify frame_number)
    - Multiple tools: brush (stroke points), touch (single/multiple points), lasso (closed path), shape (bounding box)
    - Returns contour polygon points for rendering on the frontend
    - Uses unified SAM2VideoPredictor for both images and videos with lazy frame loading
    """
    try:
        # Validate input file
        input_path, is_video = validate_input_file(request.input_path)

        # For videos, frame_number is required
        if is_video and request.frame_number is None:
            raise HTTPException(
                status_code=400, detail="frame_number is required for video inputs"
            )

        logger.info(
            f"Processing {'video' if is_video else 'image'}: {input_path}, "
            f"frame: {request.frame_number if is_video else 'N/A'}, tool: {request.tool}"
        )

        # Prepare SAM2 inputs based on tool (we need to pass dummy image shape for validation)
        # The actual image dimensions will be determined by the predictor
        inputs = prepare_mask_inputs(
            request, (1920, 1080, 3)
        )  # Dummy shape for validation

        # Validate that we have at least one input
        if inputs["point_coords"] is None and inputs["box"] is None:
            raise HTTPException(
                status_code=400,
                detail=f"No valid inputs provided for tool '{request.tool}'. Provide points or box.",
            )

        # Get SAM2 model actor
        try:
            model_type_enum = ModelType[
                os.environ.get("MASK_MODEL", "sam2_base_plus").upper()
            ]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {os.environ.get('MASK_MODEL', 'sam2_base_plus')}. Must be one of: sam2_tiny, sam2_small, sam2_base_plus, sam2_large",
            )

        predictor = get_sam2_predictor(model_type=model_type_enum)

        # Generate mask directly (seed with lasso polygon if tool is lasso)
        logger.info(f"Generating mask with unified SAM2 predictor...")
        lasso_points_flat = None
        if request.tool == "lasso" and request.points:
            # Flatten to [x1, y1, x2, y2, ...]
            lasso_points_flat = []
            for p in request.points:
                lasso_points_flat.extend([float(p["x"]), float(p["y"])])

        # For lasso, prefer seed-mask path; still pass box if present (ignored in mask path)
        point_coords = None if lasso_points_flat is not None else inputs["point_coords"]
        point_labels = None if lasso_points_flat is not None else inputs["point_labels"]

        # If a shape_type is provided and a box is present, build a binary mask for the shape
        # and seed the predictor with that mask (similar to lasso seeding)
        init_shape_mask = None
        try:
            if request.shape_type and inputs["box"] is not None:
                _ = predictor.get_predictor()
                inference_state = predictor._get_or_create_inference_state(
                    input_path=str(input_path),
                    frame_number=request.frame_number,
                    id=request.id,
                )
                from src.mask.mask import _bounds_from_box_and_shape, _rasterize_shape

                bounds = _bounds_from_box_and_shape(inputs["box"], request.shape_type)
                H = int(inference_state.get("video_height"))
                W = int(inference_state.get("video_width"))
                init_shape_mask = _rasterize_shape(
                    bounds, request.shape_type, height=H, width=W
                )
        except Exception as e:
            logger.warning(f"Failed to build shape seed mask: {e}")

        contours, mask = predictor.predict_mask(
            id=request.id,
            input_path=str(input_path),
            frame_number=request.frame_number,
            point_coords=point_coords if init_shape_mask is None else None,
            point_labels=point_labels if init_shape_mask is None else None,
            box=inputs["box"] if init_shape_mask is None else None,
            obj_id=1,
            simplify_tolerance=request.simplify_tolerance,
            lasso_points=lasso_points_flat,
            init_mask=init_shape_mask,
        )

        logger.info(f"Mask generated successfully: {len(contours)} contours")

        # Save debug visualizations if requested
        if request.debug:
            logger.info("Debug mode enabled - saving visualizations...")
            save_debug_visualizations(
                input_path=input_path,
                frame_number=request.frame_number,
                point_coords=inputs["point_coords"],
                point_labels=inputs["point_labels"],
                box=inputs["box"],
                contours=contours,
                mask=mask,
                seed_mask=init_shape_mask,
            )

        return MaskResponse(
            status="success",
            contours=contours,
            message=f"Generated mask with {len(contours)} contour(s)",
            input_path=str(input_path),
            frame_number=request.frame_number if is_video else None,
            tool=request.tool,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to generate mask: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

        return MaskResponse(
            status="error",
            message=f"Failed to generate mask: {str(e)}",
            input_path=request.input_path,
            frame_number=request.frame_number,
            tool=request.tool,
        )


@router.post("/track")
async def track_mask(request: MaskTrackingRequest):
    """
    Stream propagated mask contours for a prior touch-point mask across a frame range.
    Uses existing SAM2 inference state keyed by id and the first completed frame, and
    propagates via SAM2's propagate_in_video. Emits NDJSON lines: {frame_number, contours}.
    """
    try:
        # Validate input file
        input_path, is_video = validate_input_file(request.input_path)
        if not is_video:
            raise HTTPException(
                status_code=400, detail="Mask tracking only supports video inputs"
            )
        if request.frame_start is None or request.frame_end is None:
            raise HTTPException(
                status_code=400, detail="frame_start and frame_end are required"
            )

        # Validate model type
        try:
            model_type_enum = ModelType[
                os.environ.get("MASK_MODEL", "sam2_base_plus").upper()
            ]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model_type: {os.environ.get('MASK_MODEL', 'sam2_base_plus')}",
            )

        predictor = get_sam2_predictor(model_type=model_type_enum)

        # Resolve direction
        direction = request.direction or (
            "forward" if request.frame_end >= request.frame_start else "backward"
        )

        # Validate ranges for one-way cases
        if direction == "forward" and request.frame_end < request.frame_start:
            raise HTTPException(
                status_code=400,
                detail="For forward tracking, frame_end must be >= frame_start",
            )
        if direction == "backward" and request.frame_end > request.frame_start:
            raise HTTPException(
                status_code=400,
                detail="For backward tracking, frame_end must be <= frame_start",
            )

        anchor = (
            request.anchor_frame
            if request.anchor_frame is not None
            else request.frame_start
        )

        # Ensure old cancel flags for this id are cleared before starting
        try:
            CANCEL_TRACKING.discard(request.id)
        except Exception:
            pass

        # Single-direction streaming
        if direction in ("forward", "backward"):

            def ndjson_generator_single():
                try:
                    for item in predictor.iter_track_masks(
                        input_path=str(input_path),
                        frame_start=int(request.frame_start),
                        frame_end=int(request.frame_end),
                        anchor_frame=int(anchor),
                        max_frames=request.max_frames,
                        id=request.id,
                        simplify_tolerance=1.0,
                    ):
                        # Cooperative cancellation check
                        if request.id in CANCEL_TRACKING:
                            # Clear flag and emit a final cancelled status
                            CANCEL_TRACKING.discard(request.id)
                            yield json.dumps({"status": "cancelled"}) + "\n"
                            break
                        yield json.dumps(item) + "\n"
                except Exception as e:
                    logger.error(f"Streaming error in track_mask: {e}")
                    yield json.dumps({"status": "error", "error": str(e)}) + "\n"
                finally:
                    # Best-effort cleanup
                    try:
                        CANCEL_TRACKING.discard(request.id)
                    except Exception:
                        pass
                    try:
                        predictor.clear_states_for_id(request.id)
                    except Exception as e:
                        logger.warning(
                            f"Predictor state clear failed after track_mask: {e}"
                        )

            return StreamingResponse(
                ndjson_generator_single(), media_type="application/x-ndjson"
            )

        # direction == both
        low = min(request.frame_start, anchor)
        high = max(anchor, request.frame_end)

        def ndjson_generator_both():
            try:
                # backward first if applicable
                for item in predictor.iter_track_masks(
                    input_path=str(input_path),
                    frame_start=int(anchor),
                    frame_end=low,
                    anchor_frame=anchor,
                    max_frames=request.max_frames,
                    id=request.id,
                ):
                    if request.id in CANCEL_TRACKING:
                        CANCEL_TRACKING.discard(request.id)
                        yield json.dumps({"status": "cancelled"}) + "\n"
                        break
                    yield json.dumps(item) + "\n"
                # forward next if applicable
                for item in predictor.iter_track_masks(
                    input_path=str(input_path),
                    frame_start=int(anchor),
                    frame_end=high,
                    anchor_frame=int(anchor),
                    max_frames=request.max_frames,
                    id=request.id,
                ):
                    if request.id in CANCEL_TRACKING:
                        CANCEL_TRACKING.discard(request.id)
                        yield json.dumps({"status": "cancelled"}) + "\n"
                        break
                    yield json.dumps(item) + "\n"
            except Exception as e:
                logger.error(f"Streaming error in track_mask (both): {e}")
                yield json.dumps({"status": "error", "error": str(e)}) + "\n"
            finally:
                try:
                    CANCEL_TRACKING.discard(request.id)
                except Exception:
                    pass
                try:
                    predictor.clear_states_for_id(request.id)
                except Exception as e:
                    logger.warning(
                        f"Predictor state clear failed after track_mask (both): {e}"
                    )

        return StreamingResponse(
            ndjson_generator_both(), media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track mask: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to track mask: {str(e)}")


@router.post("/track/shapes")
async def track_shapes(request: MaskTrackingRequest):
    """
    Stream per-frame oriented rectangle bounds based on propagated masks for an existing id.
    Emits NDJSON lines: {frame_number, shapeBounds}.
    """
    try:
        input_path, is_video = validate_input_file(request.input_path)
        if not is_video:
            raise HTTPException(
                status_code=400, detail="Shape tracking only supports video inputs"
            )
        if request.frame_start is None or request.frame_end is None:
            raise HTTPException(
                status_code=400, detail="frame_start and frame_end are required"
            )

        try:
            model_type_enum = ModelType[request.model_type.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400, detail=f"Invalid model_type: {request.model_type}"
            )

        predictor = get_sam2_predictor(model_type=model_type_enum)
        direction = request.direction or (
            "forward" if request.frame_end >= request.frame_start else "backward"
        )
        if direction == "forward" and request.frame_end < request.frame_start:
            raise HTTPException(
                status_code=400,
                detail="For forward tracking, frame_end must be >= frame_start",
            )
        if direction == "backward" and request.frame_end > request.frame_start:
            raise HTTPException(
                status_code=400,
                detail="For backward tracking, frame_end must be <= frame_start",
            )

        anchor = (
            request.anchor_frame
            if request.anchor_frame is not None
            else request.frame_start
        )

        try:
            CANCEL_TRACKING.discard(request.id)
        except Exception:
            pass

        if direction in ("forward", "backward"):

            def ndjson_generator_single():
                results = []
                try:
                    for item in predictor.iter_track_shapes(
                        input_path=str(input_path),
                        frame_start=int(request.frame_start),
                        frame_end=int(request.frame_end),
                        anchor_frame=int(anchor),
                        max_frames=request.max_frames,
                        id=request.id,
                        shape_type=request.shape_type,
                    ):
                        if request.id in CANCEL_TRACKING:
                            CANCEL_TRACKING.discard(request.id)
                            yield json.dumps({"status": "cancelled"}) + "\n"
                            break
                        yield json.dumps(item) + "\n"
                        results.append(item)
                except Exception as e:
                    logger.error(f"Streaming error in track_shapes: {e}")
                    yield json.dumps({"status": "error", "error": str(e)}) + "\n"
                finally:
                    try:
                        CANCEL_TRACKING.discard(request.id)
                        if request.debug:
                            debug_save_rectangles(str(input_path), results, request.id)
                    except Exception:
                        pass
                    try:
                        predictor.clear_states_for_id(request.id)
                    except Exception as e:
                        logger.warning(
                            f"Predictor state clear failed after track_shapes: {e}"
                        )

            return StreamingResponse(
                ndjson_generator_single(), media_type="application/x-ndjson"
            )

        low = min(request.frame_start, anchor)
        high = max(anchor, request.frame_end)

        def ndjson_generator_both():
            results = []
            try:
                for item in predictor.iter_track_shapes(
                    input_path=str(input_path),
                    frame_start=int(anchor),
                    frame_end=low,
                    anchor_frame=anchor,
                    max_frames=request.max_frames,
                    id=request.id,
                    shape_type=request.shape_type,
                ):
                    if request.id in CANCEL_TRACKING:
                        CANCEL_TRACKING.discard(request.id)
                        yield json.dumps({"status": "cancelled"}) + "\n"
                        break
                    yield json.dumps(item) + "\n"
                    results.append(item)
                for item in predictor.iter_track_shapes(
                    input_path=str(input_path),
                    frame_start=int(anchor),
                    frame_end=high,
                    anchor_frame=int(anchor),
                    max_frames=request.max_frames,
                    id=request.id,
                    shape_type=request.shape_type,
                ):
                    if request.id in CANCEL_TRACKING:
                        CANCEL_TRACKING.discard(request.id)
                        yield json.dumps({"status": "cancelled"}) + "\n"
                        break
                    yield json.dumps(item) + "\n"
                    results.append(item)
            except Exception as e:
                logger.error(f"Streaming error in track_shapes (both): {e}")
                yield json.dumps({"status": "error", "error": str(e)}) + "\n"
            finally:
                try:
                    CANCEL_TRACKING.discard(request.id)
                except Exception:
                    pass

                if request.debug:
                    debug_save_rectangles(str(input_path), results, request.id)
                try:
                    predictor.clear_states_for_id(request.id)
                except Exception as e:
                    logger.warning(
                        f"Predictor state clear failed after track_shapes (both): {e}"
                    )

        return StreamingResponse(
            ndjson_generator_both(), media_type="application/x-ndjson"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to track shapes: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to track shapes: {str(e)}")


@router.post("/track/cancel/{id}")
def cancel_track_mask(id: str):
    """Signal the server-side streaming iterator to stop for a given mask id."""
    try:
        CANCEL_TRACKING.add(id)
        return {"status": "ok", "id": id}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel tracking: {str(e)}"
        )


@router.get("/health")
def mask_health():
    """Check if the mask service is ready."""
    try:
        # Simple health report; model initializes on first request
        return {
            "status": "ready",
            "message": "Mask service is available; model loads on first use",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error checking mask service: {str(e)}"}
