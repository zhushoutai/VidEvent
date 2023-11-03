import json
import os

anno_file = "videoevent.json"
# 读取标注文件 统计出现频率最高的事件，并记录对应的视频
action_times = {}
with open(anno_file, "r") as f:
    anno = json.load(f)
    for name, video in anno["database"].items():
        annos = video["annotations"]
        for anno in annos:
            action = anno["label"]
            if action in action_times.keys():
                action_times[action] += 1
            else:
                action_times[action] = 1
# 去除掉None事件
action_times.pop("None")
# 按照事件出现频率排序
action_times = sorted(action_times.items(), key=lambda x: x[1], reverse=True)
# 选取出现频率最高的前5个事件的事件名
action_names = []
action_ids = {}
for action_time in action_times[:5]:
    action_names.append(action_time[0])
    action_ids[action_time[0]] = len(action_ids)
# 保存新的标注文件
new_anno = {}
new_anno["database"] = {}
# 读取标注文件，只选取出现频率最高的前5个事件的视频和标注
with open(anno_file, "r") as f:
    anno = json.load(f)
    database = anno["database"]
    for video_name, video in database.items():
        annos = video["annotations"]
        new_annos = []
        for a in annos:
            action = a["label"]
            if action in action_names:
                a["label_id"] = action_ids[action]
                new_annos.append(a)
        if len(new_annos) > 0:
            new_video_anno = {}
            new_video_anno["subset"] = video["subset"]
            new_video_anno["duration"] = video["duration"]
            new_video_anno["annotations"] = new_annos
            new_anno["database"][video_name] = new_video_anno
# 保存新的标注文件
with open("videoevent_new.json", "w") as f:
    json.dump(new_anno, f, indent=4, ensure_ascii=False)



