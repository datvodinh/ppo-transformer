#  Thông tin quan trọng
    state[0:8]: Điểm của bàn chơi

#  ACTION
    0: Bỏ lượt
    1: Lật Monster 
    2: Bỏ Monster vào hang
    3-8: Úp Monster kèm trang bị tương ứng (6 trang bị của Warrior)
    9-14: Úp Monster kèm trang bị tương ứng (6 trang bị của Rogue)

    15: Defeat Goblin
    16: Defeat Skeleton
    17: Defeat Orc
    18: Defeat Vampire
    19: Defeat Glolem
    20: Defeat Lich
    21: Defeat Demon
    22: Defeat Dragon
    23: Choose Warrior
    24: Choose Rogue

#  State

    state[0:8] : Số trận thắng, thua trên bàn chơi
    state[8:12] : Trạng thái bỏ lượt của 4 người trong round
    state[12] : Số thẻ monsters đã lật
    state[13] : Số monsters trong hang
    state[14:21] : Warrior, Warrior's equipments
        Warrior = [Knight Shield, 
                    Plate Armor, 
                    Torch, Holy Grail, 
                    Dragon Spear, 
                    Vorpal Sword]

    state[21:28] : Rogue, Rogue's equipments
        Rogue = [Buckler, 
                Mithril Armor, 
                Ring Of Power, 
                Invisibility Cloak, 
                Healing Potion, 
                Vorpal Dagger]

    state[28:36] : Monster đã xem trong phase
            28: Goblin
            29: Skeleton
            30: Orc
            31: Vampire
            32: Glolem
            33: Lich
            34: Demon
            35: Dragon
    state[36:40] : Phase 
        Phase 0: Chọn bỏ lượt hoặc lật monster
        Phase 1: Chọn bỏ vào hang hoặc bỏ thẻ
        Phase 2: Chọn quái vật tiêu diệt quái vật khi có vorpal
        Phase 3: Chọn Hero

#  ENVIRONMENT        
    env[0:8] = 0
        env[0:2]: Thắng,Thua của người chơi 1
        env[2:4]: Thắng,Thua của người chơi 2
        env[4:6]: Thắng,Thua của người chơi 3
        env[6:8]: Thắng,Thua của người chơi 4

    env[8:21]: 13 monsters
    env[21]: Sô monsters đã lật


    env[22]: Hero: 0: Warrior   1: Rogue
    env[23:29]: Hero equipments

    env[29:42] = 0: quái vật trong hang
    env[42]: Số lượng quái vật trong hang

    env[43:56] = 0: Người xem quái vật: -1: chưa lật; 0,1,2,3: người chơi

    env[56:60] = 0: Trạng thái trong round của 4 người chơi: 0: chơi, 1: bỏ lượt
    env[60] = 0: Hero Chosen: 0: Random, 1: Warrior, 2: Rogue
    env[61] = 0: Turn
    env[62] = 0: Phase


#  Info equipments
    Warrior = [ "Knight Shield: HP+3",
            "Plate Armor: HP+5",
            "Torch: Defeat Monsters with strength 3 or less",
            "Holy Grail: Defeat Monsters with even numbered strength",
            "Dragon Spear: Defeat the Dragon",
            "Vorpal Sword: Defeat one Monster that you choose before entering the Dungeon",
            "HP: 3"
            ]

    Rogue = [   "Buckler: HP+3",
                "Mithril Armor: HP+5",
                "Ring Of Power: Defeat Monsters with strength 2 or less and add their total stregth to your HP",
                "Invisibility Cloak: Defeat Monsters with stregth 6 or more",
                "Healing Potion: When you die, come back to life with your adventure's HP (once per Dungeon)",
                "Vorpal Dagger: Defeat one Monster that you choose before entering the Dungeon",
                "HP: 3"
                ]
