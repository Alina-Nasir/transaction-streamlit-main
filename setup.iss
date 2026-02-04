; Inno Setup Script for Pakistani Bank Transaction Parser
; This creates a Windows installer that bundles everything needed

#define MyAppName "Pakistani Bank Transaction Parser"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Pakistan Bank Parser Team"
#define MyAppExeName "PakistanBankParser.exe"
#define MyAppURL "https://github.com/yourusername/transaction-parser"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{A8F4B2C1-9E3D-4F7A-8B2C-1D4E5F6A7B8C}}
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

; Include documentation and batch processor configs
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "README.txt"
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "QUICKSTART.md"; DestDir: "{app}"; Flags: ignoreversion; DestName: "QUICKSTART.txt"
Source: "batch_config.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "batch_processor_service.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "batch_processor_start.py"; DestDir: "{app}"; Flags: ignoreversion

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
{ Custom installation messages and folder selection }

var
  BankSlipsFolder: string;
  FolderSelectionPage: TInputDirWizardPage;

procedure InitializeWizard();
begin
  { Create custom page for folder selection }
  FolderSelectionPage := CreateInputDirPage(wpSelectDir,
    'Bank Slips Folder',
    'Where should the batch processor monitor for new bank slips?',
    'Select a folder where you will drop bank slip images for automatic processing.' + #13#10 + #13#10 +
    'Note: The folder must already exist before you select it.',
    False,
    '');
  
  { Add a default suggestion path (but don't set it as the value) }
  FolderSelectionPage.Add('');
  FolderSelectionPage.Values[0] := ExpandConstant('{userdocs}\BankSlips\incoming');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  if CurPageID = FolderSelectionPage.ID then
  begin
    { Get the selected folder }
    BankSlipsFolder := FolderSelectionPage.Values[0];
    
    { Validate folder path }
    if BankSlipsFolder = '' then
    begin
      MsgBox('Please select a valid folder path.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    
    { Check if folder exists }
    if not DirExists(BankSlipsFolder) then
    begin
      MsgBox('The selected folder does not exist.' + #13#10 +
             'Please create the folder first and then select it.' + #13#10 +
             'Selected path: ' + BankSlipsFolder, mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigPath: string;
  ConfigContent: string;
  AppDataPath: string;
  EscapedPath: string;
  i: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { Create config directory }
    AppDataPath := ExpandConstant('{userappdata}\PakistanBankParser\config');
    ForceDirectories(AppDataPath);
    
    { Escape backslashes for JSON - replace single \ with \\ }
    EscapedPath := '';
    for i := 1 to Length(BankSlipsFolder) do
    begin
      if BankSlipsFolder[i] = '\' then
        EscapedPath := EscapedPath + '\\'
      else
        EscapedPath := EscapedPath + BankSlipsFolder[i];
    end;
    
    { Create batch configuration file with user-selected folder }
    ConfigPath := AppDataPath + '\batch_config.json';
    ConfigContent := 
      '{' + #13#10 +
      '  "incoming_folder": "' + EscapedPath + '",' + #13#10 +
      '  "debounce_delay": 3.0,' + #13#10 +
      '  "enabled": true,' + #13#10 +
      '  "inference_timeout": 300' + #13#10 +
      '}';
    
    { Write config file }
    SaveStringToFile(ConfigPath, ConfigContent, False);
    
    MsgBox('Installation complete!' + #13#10 + #13#10 + 
           'Batch Processing Configuration:' + #13#10 +
           'Incoming Folder: ' + BankSlipsFolder + #13#10 + #13#10 +
           'How it works:' + #13#10 +
           '• Drop bank slip images in the configured folder' + #13#10 +
           '• The system will automatically process them' + #13#10 +
           '• Processed files remain in the same folder' + #13#10 +
           '• Results are saved to the database' + #13#10 + #13#10 +
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
         '• All required libraries' + #13#10 +
         '• Automatic batch processing service' + #13#10 + #13#10 +
         'No internet connection required after installation!' + #13#10 + #13#10 +
         'Installation size: ~2.5 GB' + #13#10 + #13#10 +
         'You will be asked to select a folder for batch processing.',
         mbInformation, MB_OK);
  Result := True;
end;

