/**
 * Windows Code Signing Script for Apex Studio
 * 
 * This script handles code signing for Windows builds.
 * It supports both PFX file signing and Azure Key Vault signing.
 * 
 * Required environment variables:
 * - WINDOWS_CERT_FILE: Path to the .pfx certificate file
 * - WINDOWS_CERT_PASSWORD: Password for the certificate
 * 
 * For Azure Key Vault (optional):
 * - AZURE_TENANT_ID
 * - AZURE_CLIENT_ID
 * - AZURE_CLIENT_SECRET
 * - AZURE_KEY_VAULT_URI
 * - AZURE_KEY_VAULT_CERT_NAME
 */

const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

// Configuration
const TIMESTAMP_SERVER = "http://timestamp.digicert.com";
const HASH_ALGORITHM = "sha256";

/**
 * Sign a Windows executable
 * @param {object} configuration - The electron-builder configuration
 */
exports.default = async function sign(configuration) {
  const { path: filePath, hash } = configuration;
  
  // Skip if no signing certificate is configured
  if (!process.env.WINDOWS_CERT_FILE && !process.env.AZURE_KEY_VAULT_URI) {
    console.log(`Skipping signing for ${filePath} - no certificate configured`);
    return;
  }

  // Get file extension
  const ext = path.extname(filePath).toLowerCase();
  
  // Only sign executables and DLLs
  if (![".exe", ".dll", ".msi"].includes(ext)) {
    console.log(`Skipping non-signable file: ${filePath}`);
    return;
  }

  console.log(`Signing: ${filePath}`);

  try {
    if (process.env.AZURE_KEY_VAULT_URI) {
      await signWithAzureKeyVault(filePath);
    } else {
      await signWithSignTool(filePath);
    }
    console.log(`Successfully signed: ${filePath}`);
  } catch (error) {
    console.error(`Failed to sign ${filePath}:`, error.message);
    
    // In CI, fail the build on signing errors
    if (process.env.CI) {
      throw error;
    }
  }
};

/**
 * Sign using Windows SignTool with a PFX certificate
 */
async function signWithSignTool(filePath) {
  const certFile = process.env.WINDOWS_CERT_FILE;
  const certPassword = process.env.WINDOWS_CERT_PASSWORD;

  if (!certFile || !fs.existsSync(certFile)) {
    throw new Error(`Certificate file not found: ${certFile}`);
  }

  // Find SignTool
  const signToolPath = findSignTool();

  // Build the command
  const args = [
    "sign",
    "/f", `"${certFile}"`,
    "/p", `"${certPassword}"`,
    "/tr", TIMESTAMP_SERVER,
    "/td", HASH_ALGORITHM,
    "/fd", HASH_ALGORITHM,
    "/d", '"Apex Studio"',
    "/du", '"https://apex.studio"',
    `"${filePath}"`,
  ];

  execSync(`"${signToolPath}" ${args.join(" ")}`, {
    stdio: "inherit",
    windowsHide: true,
  });
}

/**
 * Sign using Azure Key Vault (for CI/CD environments)
 */
async function signWithAzureKeyVault(filePath) {
  const tenantId = process.env.AZURE_TENANT_ID;
  const clientId = process.env.AZURE_CLIENT_ID;
  const clientSecret = process.env.AZURE_CLIENT_SECRET;
  const keyVaultUri = process.env.AZURE_KEY_VAULT_URI;
  const certName = process.env.AZURE_KEY_VAULT_CERT_NAME;

  // Use AzureSignTool for Key Vault signing
  // Install: dotnet tool install --global AzureSignTool
  const args = [
    "sign",
    "-kvu", keyVaultUri,
    "-kvi", clientId,
    "-kvt", tenantId,
    "-kvs", clientSecret,
    "-kvc", certName,
    "-tr", TIMESTAMP_SERVER,
    "-td", HASH_ALGORITHM,
    "-fd", HASH_ALGORITHM,
    "-d", '"Apex Studio"',
    "-du", '"https://apex.studio"',
    `"${filePath}"`,
  ];

  execSync(`AzureSignTool ${args.join(" ")}`, {
    stdio: "inherit",
    windowsHide: true,
  });
}

/**
 * Find the Windows SDK SignTool
 */
function findSignTool() {
  const windowsKitPaths = [
    "C:\\Program Files (x86)\\Windows Kits\\10\\bin",
    "C:\\Program Files\\Windows Kits\\10\\bin",
  ];

  for (const kitPath of windowsKitPaths) {
    if (!fs.existsSync(kitPath)) continue;

    // Find the latest version
    const versions = fs.readdirSync(kitPath)
      .filter((v) => v.match(/^\d+\.\d+\.\d+\.\d+$/))
      .sort()
      .reverse();

    for (const version of versions) {
      const signToolPath = path.join(kitPath, version, "x64", "signtool.exe");
      if (fs.existsSync(signToolPath)) {
        return signToolPath;
      }
    }
  }

  // Try to find in PATH
  try {
    execSync("where signtool", { stdio: "ignore" });
    return "signtool";
  } catch {
    throw new Error("SignTool not found. Please install Windows SDK.");
  }
}

