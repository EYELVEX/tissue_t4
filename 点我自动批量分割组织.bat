@echo off
setlocal enabledelayedexpansion

:: Define input and output directories
set "input_dir=imagesTr"
set "output_dir=labelsTr_tissue"

:: Create output directory if it doesn't exist
if not exist "%output_dir%" mkdir "%output_dir%"

:: Loop through all .nii.gz files in input directory
for %%F in ("%input_dir%\*.nii.gz") do (
    set "filename=%%~nxF"
    set "input_path=%input_dir%\!filename!"
    set "output_path=%output_dir%\!filename!"
    echo Processing: !filename!
    TotalSegmentator -i "!input_path!" -o "!output_path!" --ml -ta tissue_types )

echo Done.
pause