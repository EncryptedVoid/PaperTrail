
using namespace System.Management.Automation
using namespace System.Management.Automation.Language

Register-ArgumentCompleter -Native -CommandName 'ouch' -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)

    $commandElements = $commandAst.CommandElements
    $command = @(
        'ouch'
        for ($i = 1; $i -lt $commandElements.Count; $i++) {
            $element = $commandElements[$i]
            if ($element -isnot [StringConstantExpressionAst] -or
                $element.StringConstantType -ne [StringConstantType]::BareWord -or
                $element.Value.StartsWith('-') -or
                $element.Value -eq $wordToComplete) {
                break
        }
        $element.Value
    }) -join ';'

    $completions = @(switch ($command) {
        'ouch' {
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help (see more with ''--help'')')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help (see more with ''--help'')')
            [CompletionResult]::new('-V', '-V ', [CompletionResultType]::ParameterName, 'Print version')
            [CompletionResult]::new('--version', '--version', [CompletionResultType]::ParameterName, 'Print version')
            [CompletionResult]::new('compress', 'compress', [CompletionResultType]::ParameterValue, 'Compress one or more files into one output file')
            [CompletionResult]::new('c', 'c', [CompletionResultType]::ParameterValue, 'Compress one or more files into one output file')
            [CompletionResult]::new('decompress', 'decompress', [CompletionResultType]::ParameterValue, 'Decompresses one or more files, optionally into another folder')
            [CompletionResult]::new('d', 'd', [CompletionResultType]::ParameterValue, 'Decompresses one or more files, optionally into another folder')
            [CompletionResult]::new('list', 'list', [CompletionResultType]::ParameterValue, 'List contents of an archive')
            [CompletionResult]::new('l', 'l', [CompletionResultType]::ParameterValue, 'List contents of an archive')
            [CompletionResult]::new('ls', 'ls', [CompletionResultType]::ParameterValue, 'List contents of an archive')
            [CompletionResult]::new('help', 'help', [CompletionResultType]::ParameterValue, 'Print this message or the help of the given subcommand(s)')
            break
        }
        'ouch;compress' {
            [CompletionResult]::new('-l', '-l', [CompletionResultType]::ParameterName, 'Compression level, applied to all formats')
            [CompletionResult]::new('--level', '--level', [CompletionResultType]::ParameterName, 'Compression level, applied to all formats')
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--fast', '--fast', [CompletionResultType]::ParameterName, 'Fastest compression level possible, conflicts with --level and --slow')
            [CompletionResult]::new('--slow', '--slow', [CompletionResultType]::ParameterName, 'Slowest (and best) compression level possible, conflicts with --level and --fast')
            [CompletionResult]::new('-S', '-S ', [CompletionResultType]::ParameterName, 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)')
            [CompletionResult]::new('--follow-symlinks', '--follow-symlinks', [CompletionResultType]::ParameterName, 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;c' {
            [CompletionResult]::new('-l', '-l', [CompletionResultType]::ParameterName, 'Compression level, applied to all formats')
            [CompletionResult]::new('--level', '--level', [CompletionResultType]::ParameterName, 'Compression level, applied to all formats')
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--fast', '--fast', [CompletionResultType]::ParameterName, 'Fastest compression level possible, conflicts with --level and --slow')
            [CompletionResult]::new('--slow', '--slow', [CompletionResultType]::ParameterName, 'Slowest (and best) compression level possible, conflicts with --level and --fast')
            [CompletionResult]::new('-S', '-S ', [CompletionResultType]::ParameterName, 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)')
            [CompletionResult]::new('--follow-symlinks', '--follow-symlinks', [CompletionResultType]::ParameterName, 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;decompress' {
            [CompletionResult]::new('-d', '-d', [CompletionResultType]::ParameterName, 'Place results in a directory other than the current one')
            [CompletionResult]::new('--dir', '--dir', [CompletionResultType]::ParameterName, 'Place results in a directory other than the current one')
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-r', '-r', [CompletionResultType]::ParameterName, 'Remove the source file after successful decompression')
            [CompletionResult]::new('--remove', '--remove', [CompletionResultType]::ParameterName, 'Remove the source file after successful decompression')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;d' {
            [CompletionResult]::new('-d', '-d', [CompletionResultType]::ParameterName, 'Place results in a directory other than the current one')
            [CompletionResult]::new('--dir', '--dir', [CompletionResultType]::ParameterName, 'Place results in a directory other than the current one')
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-r', '-r', [CompletionResultType]::ParameterName, 'Remove the source file after successful decompression')
            [CompletionResult]::new('--remove', '--remove', [CompletionResultType]::ParameterName, 'Remove the source file after successful decompression')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;list' {
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-t', '-t', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('--tree', '--tree', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;l' {
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-t', '-t', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('--tree', '--tree', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;ls' {
            [CompletionResult]::new('-f', '-f', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('--format', '--format', [CompletionResultType]::ParameterName, 'Specify the format of the archive')
            [CompletionResult]::new('-p', '-p', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('--password', '--password', [CompletionResultType]::ParameterName, 'Decompress or list with password')
            [CompletionResult]::new('-c', '-c', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('--threads', '--threads', [CompletionResultType]::ParameterName, 'Concurrent working threads')
            [CompletionResult]::new('-t', '-t', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('--tree', '--tree', [CompletionResultType]::ParameterName, 'Show archive contents as a tree')
            [CompletionResult]::new('-y', '-y', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('--yes', '--yes', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to yes')
            [CompletionResult]::new('-n', '-n', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('--no', '--no', [CompletionResultType]::ParameterName, 'Skip [Y/n] questions, default to no')
            [CompletionResult]::new('-A', '-A ', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('--accessible', '--accessible', [CompletionResultType]::ParameterName, 'Activate accessibility mode, reducing visual noise')
            [CompletionResult]::new('-H', '-H ', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('--hidden', '--hidden', [CompletionResultType]::ParameterName, 'Ignore hidden files')
            [CompletionResult]::new('-q', '-q', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('--quiet', '--quiet', [CompletionResultType]::ParameterName, 'Silence output')
            [CompletionResult]::new('-g', '-g', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('--gitignore', '--gitignore', [CompletionResultType]::ParameterName, 'Ignore files matched by git''s ignore files')
            [CompletionResult]::new('-h', '-h', [CompletionResultType]::ParameterName, 'Print help')
            [CompletionResult]::new('--help', '--help', [CompletionResultType]::ParameterName, 'Print help')
            break
        }
        'ouch;help' {
            [CompletionResult]::new('compress', 'compress', [CompletionResultType]::ParameterValue, 'Compress one or more files into one output file')
            [CompletionResult]::new('decompress', 'decompress', [CompletionResultType]::ParameterValue, 'Decompresses one or more files, optionally into another folder')
            [CompletionResult]::new('list', 'list', [CompletionResultType]::ParameterValue, 'List contents of an archive')
            [CompletionResult]::new('help', 'help', [CompletionResultType]::ParameterValue, 'Print this message or the help of the given subcommand(s)')
            break
        }
        'ouch;help;compress' {
            break
        }
        'ouch;help;decompress' {
            break
        }
        'ouch;help;list' {
            break
        }
        'ouch;help;help' {
            break
        }
    })

    $completions.Where{ $_.CompletionText -like "$wordToComplete*" } |
        Sort-Object -Property ListItemText
}
