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
Source: "validate_mssql.bat"; DestDir: "{app}"; Flags: ignoreversion

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
{ Custom installation with MS SQL Server configuration and folder selection }

var
  BankSlipsFolder: string;
  FolderSelectionPage: TInputDirWizardPage;
  MSSQLConfigPage: TInputQueryWizardPage;
  AuthTypePage: TInputOptionWizardPage;
  MSSQLHost, MSSQLPort, MSSQLDatabase, MSSQLUser, MSSQLPassword: string;
  UseWindowsAuth: Boolean;

procedure InitializeWizard();
begin
  { Create authentication type selection page }
  AuthTypePage := CreateInputOptionPage(wpSelectDir,
    'MS SQL Server Authentication',
    'Choose authentication method',
    'Select how you want to connect to MS SQL Server:',
    True, False);
  
  AuthTypePage.Add('Windows Authentication (Recommended)');
  AuthTypePage.Add('SQL Server Authentication (username/password)');
  AuthTypePage.Values[0] := True;  { Default to Windows Authentication }
  
  { Create custom page for MS SQL configuration - after auth type }
  MSSQLConfigPage := CreateInputQueryPage(AuthTypePage.ID,
    'MS SQL Server Configuration',
    'Configure MS SQL Server connection for storing transaction data',
    'Enter your SQL Server details. Port 1433 will be used by default.' + #13#10 + #13#10 +
    'Note: SQL Server must be running.');
  
  { Add input fields }
  MSSQLConfigPage.Add('Host (e.g., localhost or server\instance):', False);
  MSSQLConfigPage.Add('Database Name:', False);
  MSSQLConfigPage.Add('Username (only for SQL Auth):', False);
  MSSQLConfigPage.Add('Password (only for SQL Auth):', True);  // Password field (masked)
  
  { Set default values }
  MSSQLConfigPage.Values[0] := 'localhost';
  MSSQLConfigPage.Values[1] := 'transactions_db';
  MSSQLConfigPage.Values[2] := 'sa';
  MSSQLConfigPage.Values[3] := '';
  
  { Create custom page for folder selection }
  FolderSelectionPage := CreateInputDirPage(MSSQLConfigPage.ID,
    'Bank Slips Folder',
    'Where should the batch processor monitor for new bank slips?',
    'Select a folder where you will drop bank slip images for automatic processing.' + #13#10 + #13#10 +
    'Note: The folder must already exist before you select it.',
    False,
    '');
  
  { Add a default suggestion path }
  FolderSelectionPage.Add('');
  FolderSelectionPage.Values[0] := ExpandConstant('{userdocs}\BankSlips\incoming');
end;

function ValidateMSSQLConnection(Host, Port, Database, User, Password: string): Boolean;
var
  ResultCode: Integer;
  OutputMsg: AnsiString;
  TempFile: string;
  SQLCmd: string;
begin
  Result := True;  { Default to success so installation can continue }
  
  { Create temporary file for output }
  TempFile := ExpandConstant('{tmp}\mssql_validation.txt');
  
  { Build sqlcmd command to test connection }
  if UseWindowsAuth then
  begin
    { Windows Authentication }
    SQLCmd := 'sqlcmd -S ' + Host + ',' + Port + ' -E -Q "SELECT 1" 2>&1';
  end
  else
  begin
    { SQL Server Authentication }
    SQLCmd := 'sqlcmd -S ' + Host + ',' + Port + ' -U ' + User;
    if Password <> '' then
      SQLCmd := SQLCmd + ' -P ' + Password;
    SQLCmd := SQLCmd + ' -Q "SELECT 1" 2>&1';
  end;
  
  { Try to execute sqlcmd command }
  if Exec('cmd.exe', '/c ' + SQLCmd + ' > "' + TempFile + '"',
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
  begin
    if ResultCode = 0 then
    begin
      { Connection successful, now try to create database }
      if UseWindowsAuth then
      begin
        SQLCmd := 'sqlcmd -S ' + Host + ',' + Port + ' -E';
      end
      else
      begin
        SQLCmd := 'sqlcmd -S ' + Host + ',' + Port + ' -U ' + User;
        if Password <> '' then
          SQLCmd := SQLCmd + ' -P ' + Password;
      end;
      SQLCmd := SQLCmd + ' -Q "IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = ''' + Database + ''') CREATE DATABASE [' + Database + ']"';
      
      if Exec('cmd.exe', '/c ' + SQLCmd, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      begin
        if ResultCode = 0 then
        begin
          MsgBox('MS SQL Server connection validated successfully!' + #13#10 + 
                 'Database "' + Database + '" is ready.', mbInformation, MB_OK);
          Result := True;
        end
        else
        begin
          MsgBox('MS SQL Server connection successful but could not create database.' + #13#10 + #13#10 +
                 'Please create the database manually or check user permissions.' + #13#10 + #13#10 +
                 'Configuration has been saved.', mbError, MB_OK);
          Result := True;
        end;
      end;
    end
    else
    begin
      { SQL Server command failed - connection issue }
      if LoadStringFromFile(TempFile, OutputMsg) then
      begin
        MsgBox('MS SQL Server connection failed!' + #13#10 + #13#10 +
               'Please check:' + #13#10 +
               '• SQL Server is running' + #13#10 +
               '• Credentials are correct' + #13#10 +
               '• SQL Server allows TCP/IP connections' + #13#10 +
               '• sqlcmd is in system PATH' + #13#10 + #13#10 +
               'Configuration has been saved. You can update it later.', mbError, MB_OK);
      end;
      Result := True;
    end;
    
    { Clean up temp file }
    DeleteFile(TempFile);
  end
  else
  begin
    { Could not execute sqlcmd command }
    MsgBox('SQL Server client (sqlcmd.exe) not found in system PATH.' + #13#10 + #13#10 +
           'Configuration has been saved but could not be validated.' + #13#10 + #13#10 +
           'The application will create the database on first run if needed.', mbInformation, MB_OK);
    Result := True;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  
  { Store authentication type }
  if CurPageID = AuthTypePage.ID then
  begin
    UseWindowsAuth := AuthTypePage.Values[0];
    Result := True;
  end;
  
  { Store MS SQL configuration for validation after installation }
  if CurPageID = MSSQLConfigPage.ID then
  begin
    { Get MS SQL credentials }
    MSSQLHost := MSSQLConfigPage.Values[0];
    MSSQLPort := '1433';  { Fixed port }
    MSSQLDatabase := MSSQLConfigPage.Values[1];
    MSSQLUser := MSSQLConfigPage.Values[2];
    MSSQLPassword := MSSQLConfigPage.Values[3];
    
    { Validate inputs based on auth type }
    if (MSSQLHost = '') or (MSSQLDatabase = '') then
    begin
      MsgBox('Please fill in Host and Database Name.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    
    { For SQL Server Authentication, username is required }
    if not UseWindowsAuth then
    begin
      if MSSQLUser = '' then
      begin
        MsgBox('Username is required for SQL Server Authentication.', mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end;
    
    { Note: Validation will happen after files are installed }
    Result := True;
  end;
  
  { Validate folder selection }
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
  EscapedPassword: string;
  i: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { Create config directory }
    AppDataPath := ExpandConstant('{userappdata}\PakistanBankParser\config');
    ForceDirectories(AppDataPath);
    
    { Escape backslashes for JSON in folder path }
    EscapedPath := '';
    for i := 1 to Length(BankSlipsFolder) do
    begin
      if BankSlipsFolder[i] = '\' then
        EscapedPath := EscapedPath + '\\'
      else
        EscapedPath := EscapedPath + BankSlipsFolder[i];
    end;
    
    { Escape special characters in password for JSON }
    EscapedPassword := '';
    for i := 1 to Length(MSSQLPassword) do
    begin
      if MSSQLPassword[i] = '\' then
        EscapedPassword := EscapedPassword + '\\'
      else if MSSQLPassword[i] = '"' then
        EscapedPassword := EscapedPassword + '\"'
      else
        EscapedPassword := EscapedPassword + MSSQLPassword[i];
    end;
    
    { Create MS SQL database configuration file }
    ConfigPath := AppDataPath + '\db_config.json';
    
    if UseWindowsAuth then
    begin
      { Windows Authentication - no user/password needed }
      ConfigContent := 
        '{' + #13#10 +
        '  "type": "mssql",' + #13#10 +
        '  "host": "' + MSSQLHost + '",' + #13#10 +
        '  "port": ' + MSSQLPort + ',' + #13#10 +
        '  "database": "' + MSSQLDatabase + '",' + #13#10 +
        '  "user": "",' + #13#10 +
        '  "password": "",' + #13#10 +
        '  "trusted_connection": true' + #13#10 +
        '}';
    end
    else
    begin
      { SQL Server Authentication - use user/password }
      ConfigContent := 
        '{' + #13#10 +
        '  "type": "mssql",' + #13#10 +
        '  "host": "' + MSSQLHost + '",' + #13#10 +
        '  "port": ' + MSSQLPort + ',' + #13#10 +
        '  "database": "' + MSSQLDatabase + '",' + #13#10 +
        '  "user": "' + MSSQLUser + '",' + #13#10 +
        '  "password": "' + EscapedPassword + '",' + #13#10 +
        '  "trusted_connection": false' + #13#10 +
        '}';
    end;
    
    { Write MS SQL config file }
    SaveStringToFile(ConfigPath, ConfigContent, False);
    
    { Create batch processing configuration file }
    ConfigPath := AppDataPath + '\batch_config.json';
    ConfigContent := 
      '{' + #13#10 +
      '  "incoming_folder": "' + EscapedPath + '",' + #13#10 +
      '  "debounce_delay": 3.0,' + #13#10 +
      '  "enabled": true,' + #13#10 +
      '  "inference_timeout": 300' + #13#10 +
      '}';
    
    { Write batch config file }
    SaveStringToFile(ConfigPath, ConfigContent, False);
    
    { Validate MS SQL connection now that files are installed }
    if ValidateMSSQLConnection(MSSQLHost, MSSQLPort, MSSQLDatabase, MSSQLUser, MSSQLPassword) then
    begin
      MsgBox('Installation complete!' + #13#10 + #13#10 + 
           'MS SQL Server Configuration:' + #13#10 +
           '• Host: ' + MSSQLHost + ':' + MSSQLPort + #13#10 +
           '• Database: ' + MSSQLDatabase + #13#10 +
           '• User: ' + MSSQLUser + #13#10 + #13#10 +
           'Batch Processing Configuration:' + #13#10 +
           '• Incoming Folder: ' + BankSlipsFolder + #13#10 + #13#10 +
           'How it works:' + #13#10 +
           '• Drop bank slip images in the configured folder' + #13#10 +
           '• The system will automatically process them' + #13#10 +
           '• Processed files remain in the same folder' + #13#10 +
           '• Results are saved to MS SQL Server database' + #13#10 + #13#10 +
           'Important Notes:' + #13#10 +
           '• First launch may take 15-20 seconds to initialize' + #13#10 +
           '• The AI model will start automatically' + #13#10 +
           '• Your browser will open with the application' + #13#10 +
           '• Configuration stored in: %APPDATA%\PakistanBankParser' + #13#10 + #13#10 +
           'For help, see QUICKSTART.txt in the installation folder.',
           mbInformation, MB_OK);
    end
    else
    begin
      MsgBox('Installation complete but MS SQL Server connection validation failed!' + #13#10 + #13#10 +
             'The application is installed, but you may need to update MS SQL credentials.' + #13#10 + #13#10 +
             'Configuration file location:' + #13#10 +
             '%APPDATA%\PakistanBankParser\config\db_config.json' + #13#10 + #13#10 +
             'Please ensure:' + #13#10 +
             '• SQL Server is running' + #13#10 +
             '• Credentials are correct' + #13#10 +
             '• Database exists or user has permission to create it',
             mbError, MB_OK);
    end;
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
         '• MS SQL Server connector' + #13#10 +
         '• All required libraries' + #13#10 +
         '• Automatic batch processing service' + #13#10 + #13#10 +
         'Prerequisites:' + #13#10 +
         '• MS SQL Server must be installed and running on your system' + #13#10 +
         '• You will need SQL Server credentials to configure the database' + #13#10 + #13#10 +
         'No internet connection required after installation!' + #13#10 + #13#10 +
         'Installation size: ~2.5 GB' + #13#10 + #13#10 +
         'You will be asked to:' + #13#10 +
         '1. Configure MS SQL Server connection' + #13#10 +
         '2. Select a folder for batch processing',
         mbInformation, MB_OK);
  Result := True;
end;

