
from transformers import AutoTokenizer, AutoModelWithLMHead

def target():
    tk = AutoTokenizer.from_pretrained("huggingtweets/elonmusk",local_files_first=True)
    md =  AutoModelWithLMHead.from_pretrained("huggingtweets/elonmusk",local_files_first=True) 
    print("model loaded")

import multiprocessing
def test_main():
    p = multiprocessing.Process(target=target,daemon=False,)
    p.start()
    p.join()
