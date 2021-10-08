import pandas as pd
import subprocess
import platform
import os,sys

#copied from week 8 lectorial code
def strip_header(the_file):
  with open(the_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line
        else:
            break #stop when there are no more #
  the_header = header[1:].strip().split('\t')
  df = pd.read_csv(the_file,comment='#',names=the_header,sep='\t')
  return df


def get_combined_dataset_testset():
  dataset = strip_header('train.tsv')
  testset = strip_header('test.tsv')

  feature_engineering_dataset=strip_header('train_feature_engineering_nltk.tsv')
  combined_dataset=dataset.copy(deep=False)
  combined_dataset['frac_stop'] = feature_engineering_dataset["frac_stop"]
  combined_dataset['cover_stop'] = feature_engineering_dataset["cover_stop"]
  combined_dataset['entropy'] = feature_engineering_dataset["entropy"]

  feature_engineering_testset= strip_header('test_feature_engineering_nltk.tsv')
  combined_testset=testset.copy(deep=False)
  combined_testset['frac_stop'] = feature_engineering_testset["frac_stop"]
  combined_testset['cover_stop'] = feature_engineering_testset["cover_stop"]
  combined_testset['entropy'] = feature_engineering_testset["entropy"]
  
  return dataset, testset, combined_dataset, combined_testset

dataset, testset,combined_dataset, combined_testset = get_combined_dataset_testset()

def export_runfile(queryIDs, docIDs, predictions, filename):
    runfile = pd.DataFrame({'QueryID': list(queryIDs), 'Docid': list(docIDs), 'Score': list(predictions)}, 
                         columns=['QueryID', 'Docid', 'Score'])
    runfile2 = runfile.groupby(["QueryID"]).apply(lambda x: x.sort_values(["Score"], ascending = False)).reset_index(drop=True)
    runfile2.to_csv(filename, sep='\t', header=False, index=False)

#copied from lectorial 8 code with slight change
def get_ndcg_score(txt):
  for ln in txt.split('\n'):
    ln = ln.strip()
    fields = ln.split('\t')
    metric = fields[0].strip()
    if metric == 'ndcg':
      return fields[2].strip()

def run_trec(this_os,qrel_file,run_file,verbose):
  if this_os == 'Linux':
    tbin = './week8/trec_eval.linux'
  elif this_os == 'Windows':
    tbin = './week8/trec_eval.exe'
  elif this_os == 'Darwin':
    tbin = './week8/trec_eval.osx'
  else:
    print('OS is not known')

  try:
    args = (tbin, "-m", "all_trec",
            qrel_file, run_file)
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()
    output = popen.stdout.read()
    txt = output.decode()
  except Exception as e:
    print('[ERROR]: subprocess failed')
    print('[ERROR]: {}'.format(e))
    
  return get_ndcg_score(txt)
    
def read_sort_run(run_file):
  qdic = {}
  lines = []
  with open(run_file,'r') as f:
    for ln in f:
      ln = ln.strip()
      x,y,z = ln.split('\t')
      if x in qdic:
        qdic[x].append((y,float(z)))
      else:
        qdic[x] = []
        qdic[x].append((y,float(z)))

  rank = 1
  for k,v in qdic.items():
    v.sort(key=lambda x:x[1], reverse=True)
    rank = 1
    for a,b in v:
        out = str(k) + ' Q0 ' + a + ' ' + str(rank) + ' ' + str(b) + ' e76767'
        lines.append(out)
        rank += 1
  return '\n'.join(lines)

def calculate_ndcg(run_file):
  verbose = False
  this_os = platform.system()
  qrel_file = './week8/train.qrels'
  rf = read_sort_run(run_file)
  output_file = 'current.run'
  with open(output_file,'w') as f:
      f.write(rf)
  ndcg_score = run_trec(this_os,qrel_file,output_file,verbose)
  
  return ndcg_score