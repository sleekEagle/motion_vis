import os
data_path = r'C:\Users\lahir\Downloads\UCF101\jpgs'
dirs = os.listdir(data_path)

l = []
for dir in dirs:
    sub_path = os.path.join(data_path, dir)
    vids = os.listdir(sub_path)
    l.extend([f'{dir}/{v}' for v in vids])


server_data_path = r'/home/lahirunuwanisme/jpgs/'
server_of_path = r'/home/lahirunuwanisme/raft_flow/'
server_vis_path = r'/home/lahirunuwanisme/seg/'

out_path = 'shell.txt'

commands = []
with open(out_path, 'a') as f:
    for idx,video in enumerate(l):
        input_path = os.path.join(server_data_path, video)
        output_path = os.path.join(server_of_path, video)
        vis_path = f'{server_vis_path}{video}'
        command = f'./segment_cli/segment_cli {input_path} {output_path} --vis-dir {vis_path}'
        f.write(f"echo Processing video {idx+1}/{len(l)}\n")
        f.write(f"{command}\n")






pass