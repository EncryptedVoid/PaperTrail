# Print an optspec for argparse to handle cmd's options that are independent of any subcommand.
function __fish_ouch_global_optspecs
	string join \n y/yes n/no A/accessible H/hidden q/quiet g/gitignore f/format= p/password= c/threads= h/help V/version
end

function __fish_ouch_needs_command
	# Figure out if the current invocation already has a command.
	set -l cmd (commandline -opc)
	set -e cmd[1]
	argparse -s (__fish_ouch_global_optspecs) -- $cmd 2>/dev/null
	or return
	if set -q argv[1]
		# Also print the command, so this can be used to figure out what it is.
		echo $argv[1]
		return 1
	end
	return 0
end

function __fish_ouch_using_subcommand
	set -l cmd (__fish_ouch_needs_command)
	test -z "$cmd"
	and return 1
	contains -- $cmd[1] $argv
end

complete -c ouch -n "__fish_ouch_needs_command" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_needs_command" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_needs_command" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_needs_command" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_needs_command" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_needs_command" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_needs_command" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_needs_command" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_needs_command" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_needs_command" -s h -l help -d 'Print help (see more with \'--help\')'
complete -c ouch -n "__fish_ouch_needs_command" -s V -l version -d 'Print version'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "compress" -d 'Compress one or more files into one output file'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "c" -d 'Compress one or more files into one output file'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "decompress" -d 'Decompresses one or more files, optionally into another folder'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "d" -d 'Decompresses one or more files, optionally into another folder'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "list" -d 'List contents of an archive'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "l" -d 'List contents of an archive'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "ls" -d 'List contents of an archive'
complete -c ouch -n "__fish_ouch_needs_command" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s l -l level -d 'Compression level, applied to all formats' -r
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand compress" -l fast -d 'Fastest compression level possible, conflicts with --level and --slow'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -l slow -d 'Slowest (and best) compression level possible, conflicts with --level and --fast'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s S -l follow-symlinks -d 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand compress" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s l -l level -d 'Compression level, applied to all formats' -r
complete -c ouch -n "__fish_ouch_using_subcommand c" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand c" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand c" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand c" -l fast -d 'Fastest compression level possible, conflicts with --level and --slow'
complete -c ouch -n "__fish_ouch_using_subcommand c" -l slow -d 'Slowest (and best) compression level possible, conflicts with --level and --fast'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s S -l follow-symlinks -d 'Archive target files instead of storing symlinks (supported by `tar` and `zip`)'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand c" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s d -l dir -d 'Place results in a directory other than the current one' -r -F
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s r -l remove -d 'Remove the source file after successful decompression'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand decompress" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s d -l dir -d 'Place results in a directory other than the current one' -r -F
complete -c ouch -n "__fish_ouch_using_subcommand d" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand d" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand d" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand d" -s r -l remove -d 'Remove the source file after successful decompression'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand d" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand list" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand list" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand list" -s t -l tree -d 'Show archive contents as a tree'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand list" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand l" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand l" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand l" -s t -l tree -d 'Show archive contents as a tree'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand l" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s f -l format -d 'Specify the format of the archive' -r
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s p -l password -d 'Decompress or list with password' -r
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s c -l threads -d 'Concurrent working threads' -r
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s t -l tree -d 'Show archive contents as a tree'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s y -l yes -d 'Skip [Y/n] questions, default to yes'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s n -l no -d 'Skip [Y/n] questions, default to no'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s A -l accessible -d 'Activate accessibility mode, reducing visual noise'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s H -l hidden -d 'Ignore hidden files'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s q -l quiet -d 'Silence output'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s g -l gitignore -d 'Ignore files matched by git\'s ignore files'
complete -c ouch -n "__fish_ouch_using_subcommand ls" -s h -l help -d 'Print help'
complete -c ouch -n "__fish_ouch_using_subcommand help; and not __fish_seen_subcommand_from compress decompress list help" -f -a "compress" -d 'Compress one or more files into one output file'
complete -c ouch -n "__fish_ouch_using_subcommand help; and not __fish_seen_subcommand_from compress decompress list help" -f -a "decompress" -d 'Decompresses one or more files, optionally into another folder'
complete -c ouch -n "__fish_ouch_using_subcommand help; and not __fish_seen_subcommand_from compress decompress list help" -f -a "list" -d 'List contents of an archive'
complete -c ouch -n "__fish_ouch_using_subcommand help; and not __fish_seen_subcommand_from compress decompress list help" -f -a "help" -d 'Print this message or the help of the given subcommand(s)'
