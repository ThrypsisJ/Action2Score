import datadragon_data_manipulator as ddm

items = ddm.file_opener("item")
champions = ddm.file_opener("champion")

ddm.item_to_csv(items)
ddm.champ_to_csv(champions)