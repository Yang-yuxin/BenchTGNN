# bash run_parallel_sutlej.sh REDDIT TGAT_1l config
# bash run_parallel_sutlej.sh WIKI TGAT_1l config
# bash run_parallel_sutlej.sh MOOC TGAT_1l config
# bash run_parallel_sutlej.sh LASTFM TGAT_1l config
# bash run_parallel_sutlej.sh LASTFM mixer_wiki_1l config
# bash run_parallel_sutlej.sh MOOC mixer_wiki_1l config

# 0113
# bash run_parallel_sutlej.sh WIKI mixer_wiki_2l config
# bash run_parallel_sutlej.sh REDDIT mixer_wiki_2l config

# 0115
# bash run_parallel_sutlej.sh WIKI mixer_wiki_1l config

# bash run_parallel_sutlej.sh WIKI TGAT_1l_gru config
# bash run_parallel_sutlej.sh REDDIT TGAT_1l_gru config
# bash run_parallel_sutlej.sh REDDIT mixer_reddit_1l config

# 0116
# bash run_parallel_sutlej.sh LASTFM TGAT_1l config
# bash run_parallel_sutlej.sh MOOC TGAT_1l config

# 0117
# bash run_parallel_sutlej.sh LASTFM mixer_1l config
# bash run_parallel_sutlej.sh MOOC mixer_1l config

# 0119
# bash run_parallel_sutlej.sh LASTFM mixer_1l_fixed config
# bash run_parallel_sutlej.sh MOOC mixer_1l_fixed config

# 0122
# bash run_parallel_sutlej.sh LASTFM TGAT_1l_fixed config
# bash run_parallel_sutlej.sh MOOC TGAT_1l_fixed config

# 0122
# bash run_parallel_sutlej.sh LASTFM scan_5 config
# bash run_parallel_sutlej.sh LASTFM scan_10 config
# bash run_parallel_sutlej.sh LASTFM scan_50 config
# bash run_parallel_sutlej.sh LASTFM scan_100 config
# bash run_parallel_sutlej.sh LASTFM scan_200 config

# 0125
# bash run_parallel_sutlej.sh Flights test_time config
# bash run_parallel_sutlej.sh mooc test_time config
# bash run_parallel_sutlej.sh LASTFM test_time config
# bash run_parallel_sutlej.sh WIKI test_time config
# bash run_parallel_sutlej.sh REDDIT test_time config
# bash run_parallel_sutlej.sh CollegeMsg test_time config
# bash run_parallel_sutlej.sh uci test_time config

# bash run_parallel_sutlej.sh Flights scan_5 config
# bash run_parallel_sutlej.sh Flights scan_10 config
# bash run_parallel_sutlej.sh Flights scan_20 config
# bash run_parallel_sutlej.sh Flights scan_50 config
# bash run_parallel_sutlej.sh Flights scan_100 config
# bash run_parallel_sutlej.sh Flights scan_200 config

# bash run_parallel_ganges.sh uci scan_5x10_mixer config
# bash run_parallel_ganges.sh WIKI scan_10x5_mixer config

# bash run_parallel_ganges.sh REDDIT scan_5x10_mixer config
# bash run_parallel_ganges.sh REDDIT scan_10x5_mixer config

# bash run_parallel_ganges.sh LASTFM scan_5x10_mixer config
# bash run_parallel_ganges.sh LASTFM scan_10x5_mixer config

# bash run_parallel_ganges.sh uci scan_5x5 config
# bash run_parallel_ganges.sh uci scan_10x10 config
# bash run_parallel_ganges.sh CollegeMsg scan_5x5 config
# bash run_parallel_ganges.sh CollegeMsg scan_10x10 config

# bash run_parallel_ganges.sh WIKI scan_5_none config
# bash run_parallel_ganges.sh WIKI scan_10_none config
# bash run_parallel_ganges.sh WIKI scan_20_none config
# bash run_parallel_ganges.sh WIKI scan_50_none config
# bash run_parallel_ganges.sh WIKI scan_100_none config
# bash run_parallel_ganges.sh WIKI scan_5x5_none config
# bash run_parallel_ganges.sh WIKI scan_5x10_none config
# bash run_parallel_ganges.sh WIKI scan_10x5_none config
# bash run_parallel_ganges.sh WIKI scan_10x10_none config

# bash run_parallel_ganges.sh LASTFM scan_5_none config
# bash run_parallel_ganges.sh LASTFM scan_10_none config
# bash run_parallel_ganges.sh LASTFM scan_20_none config
# bash run_parallel_ganges.sh LASTFM scan_50_none config
# bash run_parallel_ganges.sh LASTFM scan_100_none config
# bash run_parallel_ganges.sh LASTFM scan_5x5_none config
# bash run_parallel_ganges.sh LASTFM scan_5x10_none config
# bash run_parallel_ganges.sh LASTFM scan_10x5_none config
# bash run_parallel_ganges.sh LASTFM scan_10x10_none config


# bash run_parallel_ganges.sh uci scan_5_none config
# bash run_parallel_ganges.sh uci scan_10_none config
# bash run_parallel_ganges.sh uci scan_20_none config
# bash run_parallel_ganges.sh uci scan_50_none config
# bash run_parallel_ganges.sh uci scan_100_none config
# bash run_parallel_ganges.sh uci scan_5x5_none config
# bash run_parallel_ganges.sh uci scan_5x10_none config
# bash run_parallel_ganges.sh uci scan_10x5_none config
# bash run_parallel_ganges.sh uci scan_10x10_none config

# bash run_parallel_ganges.sh CollegeMsg scan_5_none config
# bash run_parallel_ganges.sh CollegeMsg scan_10_none config
# bash run_parallel_ganges.sh CollegeMsg scan_20_none config
# bash run_parallel_ganges.sh CollegeMsg scan_50_none config
# bash run_parallel_ganges.sh CollegeMsg scan_100_none config
# bash run_parallel_ganges.sh CollegeMsg scan_5x5_none config
# bash run_parallel_ganges.sh CollegeMsg scan_5x10_none config
# bash run_parallel_ganges.sh CollegeMsg scan_10x5_none config
# bash run_parallel_ganges.sh CollegeMsg scan_10x10_none config

# Run gru and embedding memory for all datasets
# bash run_parallel_ganges.sh WIKI scan_lt5 config
# bash run_parallel_ganges.sh REDDIT scan_lt5 config
# bash run_parallel_ganges.sh mooc scan_lt5 config
# bash run_parallel_ganges.sh LASTFM scan_lt5 config
# bash run_parallel_ganges_scan10.sh Flights scan_lt5 config
# bash run_parallel_ganges_scan10.sh uci scan_lt5 config
# bash run_parallel_ganges_scan10.sh CollegeMsg scan_lt5 config

# Run no memory for all datasets
# bash run_parallel_ganges_scan10.sh WIKI scan_lt5_none config
# bash run_parallel_ganges_scan10.sh REDDIT scan_lt5_none config
# bash run_parallel_ganges_scan10.sh mooc scan_lt5_none config
# bash run_parallel_ganges_scan10.sh LASTFM scan_lt5_none config
# bash run_parallel_ganges_scan10.sh Flights scan_lt5_none config
# bash run_parallel_ganges_scan10.sh uci scan_lt5_none config
# bash run_parallel_ganges_scan10.sh CollegeMsg scan_lt5_none config

# bash run_parallel_ganges_scan10.sh WIKI scan_lt5_none config
# bash run_parallel_ganges_scan10.sh REDDIT scan_lt5_none config
# bash run_parallel_ganges_scan10.sh mooc scan_lt5_none config
# bash run_parallel_ganges_scan10.sh LASTFM scan_lt5_none config
# bash run_parallel_ganges_scan10.sh Flights scan_lt5_none config
# bash run_parallel_ganges_scan10.sh uci scan_lt5_none config
# bash run_parallel_ganges_scan10.sh CollegeMsg scan_lt5_none config

# bash run_parallel_ganges.sh syn_0.6_0.4 testsyn config
# bash run_parallel_ganges.sh syn_1_0.4 testsyn config
# bash run_parallel_ganges.sh syn_0.7_0.4 testsyn config
# bash run_parallel_ganges.sh syn_0.8_0.4 testsyn config
# bash run_parallel_ganges.sh syn_0.9_0.4 testsyn config

# bash run_parallel_ganges.sh syn_0.6_0.5 testsyn config
# bash run_parallel_ganges.sh syn_1_0.5 testsyn config
# bash run_parallel_ganges.sh syn_0.7_0.5 testsyn config
# bash run_parallel_ganges.sh syn_0.8_0.5 testsyn config
# bash run_parallel_ganges.sh syn_0.9_0.5 testsyn config

# bash run_parallel_ganges.sh syn_0.6_0.6 testsyn config
# bash run_parallel_ganges.sh syn_1_0.6 testsyn config
# bash run_parallel_ganges.sh syn_0.7_0.6 testsyn config
# bash run_parallel_ganges.sh syn_0.8_0.6 testsyn config
# bash run_parallel_ganges.sh syn_0.9_0.6 testsyn config

# bash run_parallel_ganges.sh syn_0.6_0.7 testsyn config
# bash run_parallel_ganges.sh syn_1_0.7 testsyn config
# bash run_parallel_ganges.sh syn_0.7_0.7 testsyn config
# bash run_parallel_ganges.sh syn_0.8_0.7 testsyn config
# bash run_parallel_ganges.sh syn_0.9_0.7 testsyn config

bash run_parallel_ganges.sh WIKI test_embed config