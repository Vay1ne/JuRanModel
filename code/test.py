def parse_uploader_video_data(file_path):
    # 初始化一个空字典
    uploader_dict = {}

    # 打开文件并逐行读取
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行末的换行符并拆分成uploader_id和video_id
            uploader_id, video_id = line.strip().split(' ')

            # 将uploader_id和video_id转换为整数类型
            uploader_id = int(uploader_id)
            video_id = int(video_id)

            # 如果uploader_id不在字典中，则初始化一个空列表
            if uploader_id not in uploader_dict:
                uploader_dict[uploader_id] = []

            # 将video_id添加到对应的uploader_id的列表中
            uploader_dict[uploader_id].append(video_id)

    return uploader_dict


# 使用示例
file_path = '../data/takatak/uploader_video_data.txt'
uploader_video_dict = parse_uploader_video_data(file_path)
print(uploader_video_dict)
