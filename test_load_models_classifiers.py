# whatever. let's load models.
# something in our mind: PPO, supervised, setfit as RM, language models which can take actions (webgpt-like), autonomous learning (seek goals itself), human interface automation (check terminal and display, control mouse, keyboard), assistive language models, AGI (maybe later. let's first use the LM to understand the doc and blog!)

# you want to write this in pytest compatible tests? cause in that way you don't have to write try...except every time.

# said "capsys" is for automatically simulating input and capturing output

from transformers import AutoTokenizer, AutoModelForCausalLM
# saying you should use process in order to free cuda memory. glad to know that!
#shared_objects = {}

import multiprocessing


def getVRAMUsage():
    from gpuinfo import GPUInfo
    memusage = GPUInfo.gpu_usage()
    mdict = dict(zip(*memusage))
    # this key is unstable. it cannot always get the first GPU card.
    print("dict:", mdict)
    data = mdict[list(mdict.keys())[0]]
    print("VRAM?", data, type(data))  # 1549. around.
    return data


def test_gpu_usage():
    getVRAMUsage()

# multiprocessing.set_start_method('spawn')


def load_gpt_neo():
    # https://github.com/huggingface/transformers/issues/2867
    # pull request: https://github.com/huggingface/transformers/pull/2930
    # now let's test some new shit.
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M"                                              # , local_files_only=True
                                              , local_files_first=True)
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M"    # , local_files_only=True
                                                 , local_files_first=True)
    # hey you put this shit to cuda.
    model.to('cuda')
    # quit not defined?
    # works?
    # input("about to unload.")
    assert abs(3000 - getVRAMUsage()) < 400
    # for this process we do not have stdin!
    print("loaded. exiting!")


def test_load_model():
    # how to unload the model?
    # this needs to load from somewhere. but the connection is taking too long.
    # this is fucked. someone has made some change to the long-going problem of outgoing connections?

    # it's called "resolved_archive_file" pointing to our model cache. where is it? can we invoke it directly?
    # model.to("cuda")
    # do that manually see if there's spike on VRAM.
    # what about tokenizer? usually tokenizer does not have to be GPU.
    # this model takes 1500MB VRAM.
    # use cuda?
    # not sure if this works?
    print('loading model')
    p = multiprocessing.Process(target=load_gpt_neo, daemon=False,)
    p.start()
    p.join()
    # and now you are good?
    # print("MODEL?",model)
    # recognized as gpt-neo.
    # now to check how much space it take.
    # move it back?
    # model.to("cpu")
    # del shared_objects['model']

    print('model unloaded please check VRAM')
    assert abs(1500 - getVRAMUsage()) < 400
    # now let's check VRAM, manually.

    # breakpoint()
    # check if memory leaks.

# alright, model loaded. what's next?
# check how to do reinforcement learning by any means.


def loadRLModel():
    # minRLHF requires three identical models to be loaded, which is crazy!
    # and it requires some manually setup "pad_token_id" which might be different in gpt-neo
    # too fucking hard to understand. let's just understand how "reward" works in this circumstance.
    ...
