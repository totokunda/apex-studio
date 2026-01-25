; NSIS Custom Installer Script for Apex Studio
; This script handles additional setup for the bundled Python API

; Needed for ${If} / ${EndIf} and x64/WOW64 detection.
!include "LogicLib.nsh"
!include "x64.nsh"

!macro customHeader
    !system "echo 'Building Apex Studio Installer'"
!macroend

!macro customInstall
    ; Ensure MSVC runtime (PyTorch dependency on Windows).
    ; If missing, torch can fail to import with "c10.dll ... dependency" errors.
    ;
    ; We embed `vc_redist.x64.exe` at build time (downloaded by scripts/ensure-vc-redist.js).
    ;
    ; IMPORTANT:
    ; - electron-builder's NSIS can run as a 32-bit process even for x64 targets, in which case:
    ;   - $SYSDIR == SysWOW64 (32-bit DLLs)
    ;   - but our bundled Python is x64, which needs the x64 VC runtime in the *real* System32.
    ;   So we must check the x64 runtime (registry + real System32 path via Sysnative on WOW64).

    ; 1) Prefer registry detection for VC++ 2015-2022 (14.x) x64 runtime.
    StrCpy $1 0
    ${If} ${RunningX64}
        SetRegView 64
        ClearErrors
        ReadRegDWORD $3 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
        ${IfNot} ${Errors}
            ${If} $3 == 1
                StrCpy $1 1
            ${EndIf}
        ${EndIf}
        ; Restore default registry view for the rest of the installer (important for 32-bit NSIS on x64).
        ${If} ${IsWow64}
            SetRegView 32
        ${EndIf}
    ${EndIf}

    ${If} $1 == 1
        Goto vc_redist_done
    ${EndIf}

    ; 2) Fallback: check the common trio of DLLs in the x64 System32 directory.
    StrCpy $8 "$SYSDIR"
    ${If} ${IsWow64}
        StrCpy $8 "$WINDIR\Sysnative"
    ${EndIf}

    ; $2 == 1 means "missing at least one required DLL"
    StrCpy $2 0
    IfFileExists "$8\msvcp140.dll" +2 0
        StrCpy $2 1
    IfFileExists "$8\vcruntime140.dll" +2 0
        StrCpy $2 1
    IfFileExists "$8\vcruntime140_1.dll" +2 0
        StrCpy $2 1

    StrCmp $2 0 vc_redist_done
        DetailPrint "Installing Microsoft Visual C++ Redistributable (x64) ..."
        SetOutPath "$TEMP"
        ; electron-builder defines BUILD_RESOURCES_DIR for NSIS builds; use it to reference our embedded file.
        File /oname=vc_redist.x64.exe "${BUILD_RESOURCES_DIR}\vc_redist.x64.exe"
        ExecWait '"$TEMP\vc_redist.x64.exe" /install /quiet /norestart' $0
        Delete "$TEMP\vc_redist.x64.exe"

        ; Treat success (0), reboot-required (3010), and "already installed" (1638) as OK.
        StrCmp $0 0 vc_redist_done
        StrCmp $0 3010 vc_redist_done
        StrCmp $0 1638 vc_redist_done
        MessageBox MB_ICONEXCLAMATION|MB_OK "Microsoft Visual C++ Redistributable install failed (code $0). Apex Studio may not start until it is installed."
    vc_redist_done:

    ; Post-check (best-effort): if the x64 runtime still appears missing, warn early so users don't hit torch import crashes.
    StrCpy $4 0
    ${If} ${RunningX64}
        SetRegView 64
        ClearErrors
        ReadRegDWORD $5 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
        ${IfNot} ${Errors}
            ${If} $5 == 1
                StrCpy $4 1
            ${EndIf}
        ${EndIf}
        ; Restore default registry view for the rest of the installer (important for 32-bit NSIS on x64).
        ${If} ${IsWow64}
            SetRegView 32
        ${EndIf}
    ${EndIf}

    ${If} $4 == 0
        StrCpy $6 0
        IfFileExists "$8\msvcp140.dll" +2 0
            StrCpy $6 1
        IfFileExists "$8\vcruntime140.dll" +2 0
            StrCpy $6 1
        IfFileExists "$8\vcruntime140_1.dll" +2 0
            StrCpy $6 1
        StrCmp $6 0 +2 0
            MessageBox MB_ICONEXCLAMATION|MB_OK "Warning: Visual C++ runtime DLLs (x64) still appear missing after installation. If Apex Studio fails to start with a PyTorch DLL error, install vc_redist.x64.exe manually."
    ${EndIf}
    
    ; Check for existing Python installations that might conflict
    ; Note: We use a bundled Python, so this is mainly informational
    
    ; Set environment variables for the application
    WriteRegStr HKCU "Environment" "APEX_STUDIO_PATH" "$INSTDIR"
    
    ; Create cache directories
    CreateDirectory "$LOCALAPPDATA\Apex Studio\cache"
    CreateDirectory "$LOCALAPPDATA\Apex Studio\components"
    CreateDirectory "$LOCALAPPDATA\Apex Studio\logs"
    
    ; Notify shell of environment changes
    SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
!macroend

!macro customUnInstall
    ; Clean up environment variables
    DeleteRegValue HKCU "Environment" "APEX_STUDIO_PATH"
    
    ; Notify shell of environment changes
    SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
    
    ; Ask user if they want to remove cached data
    MessageBox MB_YESNO "Do you want to remove cached models and components? This will free up disk space but you'll need to download them again." IDYES removeCache IDNO skipCache
    
    removeCache:
        RMDir /r "$LOCALAPPDATA\Apex Studio"
        Goto done
    
    skipCache:
        ; Only remove logs
        RMDir /r "$LOCALAPPDATA\Apex Studio\logs"
        
    done:
!macroend

