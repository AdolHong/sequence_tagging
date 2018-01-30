# sequence_tagging
Named Entity Recognition (biLSTM + CRF) - PyTorch

部署任务式聊天机器人时，需要关心的nlp问题有intent classification, named entity recogtion 和 dialog state tracking等。 

rasa.ai开源的聊天框架rasa_nlu与rasa_core实现了上述的三个问题，提供了我们很好的部署任务式机器人的工具。rasa官网提供的demo是英文机器人，而github也有人开源了中文支持，如[crownpku/Rasa_NLU_Chi](https://github.com/crownpku/Rasa_NLU_Chi)。

命名实体识别任务 (Named entity recognition, NER）的作用是从文本中抽取人名、地点名、公司名等信息，以填充特定任务所需要的词槽(slot-filling)。不同的任务、不同的场景，需要的词槽信息往往是不一样的。比如，查机票场景，机器人需要起点、终点、日期这三个词槽，而问天气场景，可能需要地点、日期这两个词槽。

由于rasa_nlu_chi项目中的ner部分是基于mitie实现，工作中我发现训练时间太久，于是，我用bilstm + crf模型实现了NER部分，并将代码分享。


ps: 第一次上传代码，后续继续补充readme.md