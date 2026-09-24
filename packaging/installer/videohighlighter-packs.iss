; VideoHighlighter setup with PyTorch as a pack (Inno Setup 6.1+).
;
; A real installer: Inno compresses and installs the files itself, so there is
; no .7z inside and nothing to download before the app works. What it carries:
;
;   * the core app (PyInstaller's dist/VideoHighlighter, built without torch);
;   * the CPU PyTorch pack, so every machine has a working app offline.
;
; What it can fetch, each a checkbox, downloaded from PackBase during setup
; and verified against its SHA-256:
;
;   * the NVIDIA pack (torch built for CUDA, ~4.4 GB installed) - ticked when
;     an NVIDIA card is found. It replaces the CPU pack only once it is fully
;     unpacked. How packs load: packaging/packs/rthook_packs.py.
;   * the CLIP model (visual search, chapters, taught categories) - ticked by
;     default. Model weights barely compress, so carrying it would put its
;     whole size into every download, including for people who never search.
;     It lands in {app}\models\, the first place llm/clip_prefilter.py looks.
;
; Either one failing or being declined still leaves an app that runs.
;
; Built in CI:
;   ISCC.exe /DAppVersion= /DDistDir= /DCpuPackDir= /DPackBase= \
;            /DCudaPackAsset= /DCudaPackSha256= /DCudaPackMB= /DCudaInstalledMB= \
;            /DInstalledMB= packaging\installer\videohighlighter-packs.iss
;
; FOR ME ONLY, OR FOR ALL USERS. Setup asks. "For me only" installs under
; %LOCALAPPDATA%\Programs with no admin prompt, and is the one the in-app
; updater can update in place. "For all users" installs under Program Files
; after one UAC prompt; the app runs the same (app_paths.user_data_dir moves
; its data to the user's profile when its own folder is read-only), but a
; normal user cannot write there, so updates for such an install come as a
; download (update_check.install_dir_writable) rather than in place.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef AppName
  #define AppName "VideoHighlighter"
#endif
#ifndef AppId
  #define AppId "{6E1B9C34-4F27-4A88-9D0E-1C5A7B3F8E42}"
#endif
#ifndef OutputName
  #define OutputName "VideoHighlighter-Setup"
#endif
#ifndef DistDir
  #define DistDir "..\..\dist\VideoHighlighter"
#endif
#ifndef CpuPackDir
  #define CpuPackDir "..\..\build-packs\torch-cpu"
#endif
; Where the NVIDIA pack is downloaded from, and what it must hash to.
#ifndef PackBase
  #define PackBase ""
#endif
#ifndef CudaPackAsset
  #define CudaPackAsset "component-torch-cu128.7z"
#endif
#ifndef CudaPackSha256
  #define CudaPackSha256 ""
#endif
#ifndef CudaPackMB
  #define CudaPackMB 0
#endif
#ifndef CudaInstalledMB
  #define CudaInstalledMB 0
#endif
#ifndef ClipPackAsset
  #define ClipPackAsset "component-models-clip.7z"
#endif
#ifndef ClipPackSha256
  #define ClipPackSha256 ""
#endif
#ifndef ClipPackMB
  #define ClipPackMB 0
#endif
#ifndef ClipInstalledMB
  #define ClipInstalledMB 0
#endif
#ifndef InstalledMB
  #define InstalledMB 0
#endif

#define Publisher "Aseiel"
#define AppExe "VideoHighlighter.exe"
#define CudaPackName "torch-cu128"
#define ClipDirName "clip-vit-base-patch32-ov"

[Setup]
AppId={{#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#Publisher}
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; {autopf} follows the choice: Program Files for all users, the per-user
; %LOCALAPPDATA%\Programs for me only.
DefaultDirName={autopf}\{#AppName}
UsedUserAreasWarning=no
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
OutputBaseFilename={#OutputName}
SetupIconFile=..\..\assets\icon.ico
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "nvidia"; Description: "NVIDIA GPU acceleration (downloads about {#CudaPackMB} MB)"; Flags: unchecked
Name: "clip"; Description: "Visual search model, CLIP (downloads about {#ClipPackMB} MB)"
Name: "desktopicon"; Description: "Create a &desktop shortcut"

; The bundle folder is replaced whole on every install. A build from before
; packs carried torch inside _internal, and _internal is on the frozen app's
; import path ahead of any pack: left behind, that old torch would shadow the
; pack. User data never lives there (app_paths.user_data_dir is the exe's
; folder), so nothing of the user's goes with it.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "7zr.exe"; Flags: dontcopy
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Skipped when the NVIDIA pack is already installed (Setup run again over it).
Source: "{#CpuPackDir}\*"; DestDir: "{app}\packs\torch-cpu"; Flags: ignoreversion recursesubdirs createallsubdirs; Check: NeedCpuPack

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; Packs unpacked by 7-Zip and everything the app wrote at runtime are not
; tracked by the installer, so the folder is removed whole — the same full
; removal videohighlighter.iss promises.
[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
var
  DownloadPage: TDownloadWizardPage;
  NvidiaFound: Boolean;
  NvidiaName: String;
  NvidiaDefaultApplied: Boolean;
  CudaPackDownloaded: Boolean;
  ClipPackDownloaded: Boolean;

// True when Windows reports an NVIDIA display adapter. Any failure (WMI
// disabled, a locked-down machine) is "not found": the checkbox then simply
// starts unticked, and the user can still tick it.
function HasNvidiaGpu: Boolean;
var
  Locator, Service, Items, Item: Variant;
  I: Integer;
  AdapterName: String;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Items := Service.ExecQuery('SELECT Name FROM Win32_VideoController');
    for I := 0 to Items.Count - 1 do
    begin
      Item := Items.ItemIndex(I);
      // WMI hands back a Variant; string functions need a String.
      AdapterName := Item.Name;
      Log('Display adapter: ' + AdapterName);
      if Pos('NVIDIA', Uppercase(AdapterName)) > 0 then
      begin
        Result := True;
        NvidiaName := AdapterName;
      end;
    end;
  except
    Log('GPU detection failed: ' + GetExceptionMessage);
  end;
end;

function NeedCpuPack: Boolean;
begin
  Result := not DirExists(ExpandConstant('{app}\packs\{#CudaPackName}'));
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if ProgressMax > 0 then
    DownloadPage.SetText(FileName,
      IntToStr(Progress div 1048576) + ' of ' + IntToStr(ProgressMax div 1048576) + ' MB');
  Result := True;
end;

procedure InitializeWizard;
begin
  NvidiaFound := HasNvidiaGpu;
  DownloadPage := CreateDownloadPage(
    'Downloading NVIDIA GPU acceleration',
    'PyTorch for NVIDIA cards is fetched once; app updates do not download it again.',
    @OnDownloadProgress);
end;

// Say what the detection found, so the checkbox explains itself (and a
// tester can tell detection from a wrong default at a glance).
procedure DescribeNvidiaTask;
var
  I: Integer;
begin
  for I := 0 to WizardForm.TasksList.Items.Count - 1 do
    if Pos('NVIDIA GPU acceleration', WizardForm.TasksList.ItemCaption[I]) = 1 then
    begin
      if NvidiaFound then
        WizardForm.TasksList.ItemCaption[I] := WizardForm.TasksList.ItemCaption[I]
          + ' - recommended for your ' + NvidiaName
      else
        WizardForm.TasksList.ItemCaption[I] := WizardForm.TasksList.ItemCaption[I]
          + ' - no NVIDIA card found';
      Exit;
    end;
end;

// Tick the NVIDIA box once, the first time the page is shown, when a card was
// found - after that the user's own choice stands, including on Back/Next.
procedure CurPageChanged(CurPageID: Integer);
begin
  if (CurPageID = wpSelectTasks) and not NvidiaDefaultApplied then
  begin
    DescribeNvidiaTask;
    if NvidiaFound then
      WizardSelectTasks('nvidia');
    NvidiaDefaultApplied := True;
  end;
end;

function FreeSpaceMB(const Path: String): Int64;
var
  Free, Total: Int64;
begin
  if GetSpaceOnDisk64(AddBackslash(ExtractFileDrive(Path)), Free, Total) then
    Result := Free div 1048576
  else
    Result := -1;
end;

// The install plus, with the NVIDIA pack, its download in {tmp} and its
// unpacked size. Checked before anything is fetched, so a full disk is a
// message naming both numbers instead of a failure at the end of 1.5 GB.
function SpaceLooksSufficient: Boolean;
var
  Need, Free: Int64;
  AppDir: String;
begin
  Result := True;
  Need := {#InstalledMB};
  if WizardIsTaskSelected('nvidia') then
    Need := Need + {#CudaPackMB} + {#CudaInstalledMB};
  if WizardIsTaskSelected('clip') then
    Need := Need + {#ClipPackMB} + {#ClipInstalledMB};
  if Need <= 0 then
    Exit;
  AppDir := ExpandConstant('{app}');
  Free := FreeSpaceMB(AppDir);
  if (Free >= 0) and (Free < Need) then
  begin
    MsgBox('Not enough free space on ' + ExtractFileDrive(AppDir) + '.' + #13#10#13#10
      + 'Needed: about ' + IntToStr(Need) + ' MB' + #13#10
      + 'Available: ' + IntToStr(Free) + ' MB' + #13#10#13#10
      + 'Free up space, choose a folder on another drive, or untick NVIDIA GPU acceleration.',
      mbError, MB_OK);
    Result := False;
  end;
end;

// Fetch one optional component into {tmp}. True when it arrived and matched
// its SHA-256. On failure the user decides: Yes installs without it, No sets
// Stay, which keeps the wizard on the Ready page so Next retries.
function FetchComponent(const Asset, Sha256, What, Without: String; var Stay: Boolean): Boolean;
begin
  Result := False;
  DownloadPage.Clear;
  DownloadPage.Add('{#PackBase}/' + Asset, Asset, Sha256);
  try
    DownloadPage.Download;
    Result := True;
  except
    if DownloadPage.AbortedByUser then
      Log(What + ': download cancelled by the user.')
    else
      Log(What + ': download failed: ' + GetExceptionMessage);
    Stay := SuppressibleMsgBox(
      What + ' could not be downloaded:' + #13#10
      + AddPeriod(GetExceptionMessage) + #13#10#13#10
      + 'Install without it? ' + Without,
      mbConfirmation, MB_YESNO, IDYES) <> IDYES;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Stay: Boolean;
begin
  Result := True;
  if CurPageID <> wpReady then
    Exit;

  if not SpaceLooksSufficient then
  begin
    Result := False;
    Exit;
  end;

  CudaPackDownloaded := False;
  ClipPackDownloaded := False;
  if not (WizardIsTaskSelected('nvidia') or WizardIsTaskSelected('clip')) then
    Exit;
  if '{#PackBase}' = '' then
  begin
    Log('No PackBase compiled in; nothing optional is downloaded.');
    Exit;
  end;

  Stay := False;
  DownloadPage.Show;
  try
    if WizardIsTaskSelected('nvidia') then
      CudaPackDownloaded := FetchComponent('{#CudaPackAsset}', '{#CudaPackSha256}',
        'NVIDIA GPU acceleration',
        'VideoHighlighter will run on the processor, and you can add GPU '
        + 'acceleration later by running Setup again.', Stay);
    if (not Stay) and WizardIsTaskSelected('clip') then
      ClipPackDownloaded := FetchComponent('{#ClipPackAsset}', '{#ClipPackSha256}',
        'The visual search model',
        'Everything else works; visual search, chapters and taught categories '
        + 'need it, and running Setup again adds it.', Stay);
  finally
    DownloadPage.Hide;
  end;
  Result := not Stay;
end;

// 7-Zip an archive from {tmp} into Dest. False (with the exit code logged)
// when it could not; Dest is then removed again.
function Unpack(const Asset, Dest: String): Boolean;
var
  ResultCode: Integer;
begin
  ExtractTemporaryFile('7zr.exe');
  DelTree(Dest, True, True, True);
  Result := Exec(ExpandConstant('{tmp}\7zr.exe'),
                 Format('x -y "%s" -o"%s"', [ExpandConstant('{tmp}\') + Asset, Dest]),
                 '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  if not Result then
  begin
    Log('7-Zip failed on ' + Asset + ' (exit code ' + IntToStr(ResultCode) + ')');
    DelTree(Dest, True, True, True);
  end;
end;

// Unpack the NVIDIA pack beside the CPU one, and only when that worked, remove
// the CPU pack. Until the last step the CPU pack is untouched, so any failure
// here leaves an install that starts.
procedure InstallCudaPack;
var
  Target, Staging: String;
begin
  Target := ExpandConstant('{app}\packs\{#CudaPackName}');
  Staging := Target + '.partial';
  WizardForm.StatusLabel.Caption := 'Unpacking NVIDIA GPU acceleration (this takes a few minutes)...';
  WizardForm.Refresh;

  if Unpack('{#CudaPackAsset}', Staging) then
  begin
    DelTree(Target, True, True, True);
    if RenameFile(Staging, Target) then
    begin
      DelTree(ExpandConstant('{app}\packs\torch-cpu'), True, True, True);
      Exit;
    end;
    DelTree(Staging, True, True, True);
  end;
  MsgBox('Could not install NVIDIA GPU acceleration. VideoHighlighter is '
    + 'installed and will run on the processor; run Setup again to retry.',
    mbError, MB_OK);
end;

// The archive holds models\{#ClipDirName}\...: unpack it to a staging folder
// and move that one directory into {app}\models.
procedure InstallClipPack;
var
  Staging, Target: String;
begin
  Staging := ExpandConstant('{app}\models\.clip.partial');
  Target := ExpandConstant('{app}\models\{#ClipDirName}');
  WizardForm.StatusLabel.Caption := 'Unpacking the visual search model...';
  WizardForm.Refresh;

  ForceDirectories(ExpandConstant('{app}\models'));
  if Unpack('{#ClipPackAsset}', Staging) then
  begin
    DelTree(Target, True, True, True);
    if RenameFile(Staging + '\models\{#ClipDirName}', Target) then
    begin
      DelTree(Staging, True, True, True);
      Exit;
    end;
    DelTree(Staging, True, True, True);
  end;
  MsgBox('Could not install the visual search model. Everything else works; '
    + 'run Setup again to add it.', mbError, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep <> ssPostInstall then
    Exit;
  if CudaPackDownloaded then
    InstallCudaPack;
  if ClipPackDownloaded then
    InstallClipPack;
end;
