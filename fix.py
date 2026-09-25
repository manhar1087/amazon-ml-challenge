content = open('retrieval_benchmark.py').read()
content = content.replace(
'''    return rows, cols, scores_out, gidx_out, rss_peak

for cfg in CHANNELS:''',
'''    return rows, cols, scores_out, gidx_out, rss_peak

retrieval_results = {}
for cfg in CHANNELS:'''
)
open('retrieval_benchmark.py', 'w').write(content)
