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

    ; Detect VC++ 2015-2022 (14.x) x64 runtime via registry (authoritative).
    ; NOTE: Avoid DLL presence checks here; some apps can drop DLLs without registering the runtime,
    ; which makes the "DLL exists" approach produce false positives (skipping install when still broken).
    StrCpy $1 0
    ${If} ${RunningX64}
        SetRegView 64
    ${EndIf}
    ClearErrors
    ReadRegDWORD $2 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
    ${IfNot} ${Errors}
        ${If} $2 == 1
            StrCpy $1 1
        ${EndIf}
    ${EndIf}
    ; Restore default registry view for the rest of the installer (important for 32-bit NSIS on x64).
    ${If} ${IsWow64}
        SetRegView 32
    ${EndIf}

    ; If installed already, do nothing (no prompt).
    ${If} $1 == 1
        Goto vc_redist_done
    ${EndIf}

    ; Missing: prompt user and install if they agree.
    MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON1 \
        "Apex Studio requires the Microsoft Visual C++ Redistributable (x64) to run.$\r$\n$\r$\nInstall it now?" \
        IDYES vc_redist_install IDNO vc_redist_skip

    vc_redist_install:
        DetailPrint "Installing Microsoft Visual C++ Redistributable (x64) ..."
        SetOutPath "$TEMP"
        ; electron-builder defines BUILD_RESOURCES_DIR for NSIS builds; use it to reference our embedded file.
        File /oname=vc_redist.x64.exe "${BUILD_RESOURCES_DIR}\vc_redist.x64.exe"
        ; Use passive UI so users can see what's happening; UAC may prompt for elevation.
        ExecWait '"$TEMP\vc_redist.x64.exe" /install /passive /norestart' $0
        Delete "$TEMP\vc_redist.x64.exe"

        ; Treat success (0), reboot-required (3010), and "already installed" (1638) as OK.
        StrCmp $0 0 vc_redist_postcheck
        StrCmp $0 3010 vc_redist_postcheck
        StrCmp $0 1638 vc_redist_postcheck
        MessageBox MB_ICONEXCLAMATION|MB_OK "Microsoft Visual C++ Redistributable install failed (code $0). Apex Studio may not start until it is installed."
        Goto vc_redist_done

    vc_redist_skip:
        ; User declined install; continue but warn up-front.
        MessageBox MB_ICONEXCLAMATION|MB_OK "Apex Studio may not start until the Microsoft Visual C++ Redistributable (x64) is installed."
        Goto vc_redist_done

    ; Best-effort post-check so we can warn immediately if installation didn't take.
    vc_redist_postcheck:
        StrCpy $3 0
        ${If} ${RunningX64}
            SetRegView 64
        ${EndIf}
        ClearErrors
        ReadRegDWORD $4 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
        ${IfNot} ${Errors}
            ${If} $4 == 1
                StrCpy $3 1
            ${EndIf}
        ${EndIf}
        ${If} ${IsWow64}
            SetRegView 32
        ${EndIf}
        ${If} $3 == 0
            MessageBox MB_ICONEXCLAMATION|MB_OK "Warning: Visual C++ runtime (x64) still appears missing after installation. If Apex Studio fails to start with a PyTorch DLL error, please run the installer again and accept the runtime install prompt."
        ${EndIf}

    vc_redist_done:
    
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

