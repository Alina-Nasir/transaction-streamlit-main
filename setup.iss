; Inno Setup Script for Pakistani Bank Transaction Parser
; This creates a Windows installer that bundles everything needed

#define MyAppName "Pakistani Bank Transaction Parser"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Pakistan Bank Parser Team"
#define MyAppExeName "PakistanBankParser.exe"
#define MyAppURL "https://github.com/yourusername/transaction-parser"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{A8F4B2C1-9E3D-4F7A-8B2C-1D4E5F6A7B8C}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
; Output directory for the installer
OutputDir=installer_output
OutputBaseFilename=PakistanBankParser_Setup_v{#MyAppVersion}
; Compression settings
Compression=lzma2/ultra64
SolidCompression=yes
; Windows Vista or higher required
MinVersion=6.1
; Architecture
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Privileges
PrivilegesRequired=lowest
; Icon (optional - add if you have one)
; SetupIconFile=icon.ico
; UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable
Source: "dist\PakistanBankParser\PakistanBankParser.exe"; DestDir: "{app}"; Flags: ignoreversion

; All files from _internal directory (includes Python runtime, DLLs, models, everything)
Source: "dist\PakistanBankParser\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; Include documentation
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "README.txt"
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "QUICKSTART.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "QUICKSTART.txt"

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

; Desktop shortcut (if user selected it)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Option to run the application after installation
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Clean up database from AppData on uninstall (optional)
Type: filesandordirs; Name: "{userappdata}\PakistanBankParser"

[Code]
{ Custom installation messages }
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox('Installation complete!' + #13#10 + #13#10 + 
           'Important Notes:' + #13#10 +
           '• First launch may take 15-20 seconds to initialize' + #13#10 +
           '• The AI model will start automatically' + #13#10 +
           '• Your browser will open with the application' + #13#10 +
           '• Database will be stored in: %APPDATA%\PakistanBankParser' + #13#10 + #13#10 +
           'For help, see QUICKSTART.txt in the installation folder.',
           mbInformation, MB_OK);
  end;
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  MsgBox('Welcome to Pakistani Bank Transaction Parser Setup!' + #13#10 + #13#10 +
         'This installer includes:' + #13#10 +
         '• Complete Python runtime (no Python installation needed)' + #13#10 +
         '• AI vision model (Qwen3-VL-2B, ~1.5GB)' + #13#10 +
         '• llama.cpp inference engine' + #13#10 +
         '• SQLite database' + #13#10 +
         '• All required libraries' + #13#10 + #13#10 +
         'No internet connection required after installation!' + #13#10 + #13#10 +
         'Installation size: ~2.5 GB',
         mbInformation, MB_OK);
end;
