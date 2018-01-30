# sequence_tagging
Named Entity Recognition (biLSTM + CRF) - PyTorch

部署任务式对话机器人时，需要关心的nlp问题有intent classification, named entity recogtion 和 dialog state tracking等。 

rasa.ai开源的聊天框架rasa_nlu与rasa_core，可以直接用于部署任务式对话机器人。同样地，它也是一份很好的代码，供我们学习如何解决以上几个nlp问题。

虽然，rasa官网提供的demo只支持英文，但是github上已经有中文支持的开源项目，如[crownpku/Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)。

命名实体识别任务 (Named entity recognition, NER）的作用是从文本中抽取人名、地点名、公司名等信息，以填充特定任务所需要的词槽(slot-filling)。不同的任务、不同的场景，需要的词槽信息往往是不一样的。比如，查机票场景，机器人需要起点、终点、日期这三个词槽，而问天气场景，可能需要地点、日期这两个词槽。

由于rasa_nlu_chi项目中的ner部分是基于mitie实现，工作中我发现mitie的训练时间太长。于是，我用bilstm + crf模型重新实现了NER部分，并将代码分享。 代码基于PyTorch框架。

# requirements:
Numpy 
pytorch-crf
pytorch
tensorboardX

# 代码使用
这份代码默认使用的数据是 data.json，遵从了rasa_nlu的格式。由于数据不便透漏，此处只提供了可运行的少量数据，仅供跑通代码。


训练模型: `python train.py`

测试模型: `python inference.py`

