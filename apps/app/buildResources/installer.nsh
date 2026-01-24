; NSIS Custom Installer Script for Apex Studio
; This script handles additional setup for the bundled Python API

!macro customHeader
    !system "echo 'Building Apex Studio Installer'"
!macroend

!macro customInstall
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

