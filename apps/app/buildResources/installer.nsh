; NSIS Custom Installer Script for Apex Studio
; This script handles additional setup for the bundled Python API

!macro customHeader
    !system "echo 'Building Apex Studio Installer'"
!macroend

!macro customInstall
    ; Ensure MSVC runtime (PyTorch dependency on Windows).
    ; If missing, torch can fail to import with "c10.dll ... dependency" errors.
    ;
    ; We embed `vc_redist.x64.exe` at build time (downloaded by scripts/ensure-vc-redist.js).
    IfFileExists "$SYSDIR\msvcp140.dll" vc_redist_done
        SetOutPath "$TEMP"
        ; electron-builder defines BUILD_RESOURCES_DIR for NSIS builds; use it to reference our embedded file.
        File /oname=vc_redist.x64.exe "${BUILD_RESOURCES_DIR}\vc_redist.x64.exe"
        ExecWait '"$TEMP\vc_redist.x64.exe" /install /quiet /norestart' $0
        Delete "$TEMP\vc_redist.x64.exe"

        ; Treat success (0) and reboot-required (3010) as OK.
        StrCmp $0 0 vc_redist_done
        StrCmp $0 3010 vc_redist_done
        MessageBox MB_ICONEXCLAMATION|MB_OK "Microsoft Visual C++ Redistributable install failed (code $0). Apex Studio may not start until it is installed."
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

