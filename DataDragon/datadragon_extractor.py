import datadragon_data_manipulator as ddm
import sys

version = sys.argv[1]

items = ddm.file_opener('item', version)
champions = ddm.file_opener('champion', version)

ddm.item_to_csv(items, version)
ddm.champ_to_csv(champions, version)
ddm.champ_to_dummy(version)