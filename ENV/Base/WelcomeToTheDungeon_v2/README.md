##   Thông tin quan trọng:
    [0:4]: Trạng thái bỏ lượt của mọi người
    [4:12]: Điểm của mọi người
    __STATE_SIZE__ = 71
    __ACTION_SIZE__ = 14
    __AGENT_SIZE__ = 4
    Thứ tự quái vật lần lượt là
        Golbin 1
        Golbin 2
        Skeleton 3
        Skeleton 4
        Orc 5
        Orc 6
        Vampire 7
        Vampire 8
        Golem 9
        Golem 10
        Lich 11
        Demon 12
        Dragon 13

##  :video_game: ALL_ACTION
    Trong Bidding phase:
        [0]: Bỏ lượt
        [1]: Rút thẻ quái vật và bỏ vào hang
        [2:8]: Rút thẻ quái vật và chọn 1 trang bị để bỏ đi
    Trong Dungeon phase:
        [8]: Sử dụng Vorpal Axe anh hùng Barbarian
        [9]: Sử dụng Polymorph của anh hùng Mage
        [10]: Đánh thẻ quái vật trên cùng
    Chung
        [11]: Xem bài
        [12]: Chọn Barbarian
        [13]: Chọn Mage


##  :bust_in_silhouette: P_state
    [0:4]: Trạng thái bỏ lượt của mọi người
    [4:12]: Điểm của mọi người
    [12]: Số thẻ quái vật chưa mở
    [13]: Số thẻ quái vật trong hang
    [14:53]: Các thẻ quái vật người chơi đã mở
    [53:61]: Barbarian và trang bị
        [53]: Được chọn trong vòng chơi
        [54]: Máu của Barbarian
        [55]: Leather Shield: HP+3
        [56]: Healing Potion: When Barbarian dies, comeback with 4 HP (once per dungeon)
        [57]: Torch: Defeat moster with strength 3 or less
        [58]: War Hammer: Defeat Golems
        [59]: Vorpal Axe: Defeat one monster after you draw it (once per dungeon)
        [60]: Chainmail: HP+4 
    [61:69]: Mage và trang bị
        [61]: Được chọn trong vòng chơi 
        [62]: Máu của Mage
        [63]: Omnipotence: If all the monsters in the Dungeon are different, you win the round
        [64]: Holy Grail: Defeat monster with even-numbered strength
        [65]: Demonic Pact: Defeat the Demon and the next monster
        [66]: Polymorph: Defeat one monster you draw, replacing it with the next monster from the deck (once per Dungeon)
        [67]: Wall Of Fire: HP+6
        [68]: Bracelet Of Protection: HP+3
    [69]: Trong Bidding hay Dungeon (0 là bidding)
    [70:83]: Lá quái vật nào vừa mở (1 là mở, 0 là không)
        Golbin 70
        Golbin 71
        Skeleton 72
        Skeleton 73
        Orc 74
        Orc 75
        Vampire 76
        Vampire 77
        Golem 78
        Golem 79
        Lich 80
        Demon 81
        Dragon 82


##  :globe_with_meridians: env_state
    [0:4]: Người chơi có bỏ lượt (1: có, 0: không)
    [4:12]: Thắng/Thua của người chơi
        [4:6]: Thắng,Thua của người chơi 1
        [6:8]: Thắng,Thua của người chơi 2
        [8:10]: Thắng,Thua của người chơi 3
        [10:12]: Thắng,Thua của người chơi 4
    [12:16]: Người chơi vào Dungeon Phase vòng trước (1: có, 0: không)
    [16:68]: Thông tin các thẻ quái vật 
        (sát thương | 
        ai rút, -1 là không ai rút | 
        đang trong Dungeon (thứ tự của thẻ trong Dungeon) | 
        đã bị bỏ, 1 là bỏ, 0 là ko bỏ)
        [16:20]: Golbin 1 | 0 | 0 | 0
        [20:24]: Golbin 1 | 0 | 0 | 0
        [24:28]: Skeleton 2 | 0 | 0 | 0
        [28:32]: Skeleton 2 | 0 | 0 | 0
        [32:36]: Orc 3 | 0 | 0 | 0
        [36:40]: Orc 3 | 0 | 0 | 0
        [40:44]: Vampire 4 | 0 | 0 | 0
        [44:48]: Vampire 4 | 0 | 0 | 0
        [48:52]: Golem 5 | 0 | 0 | 0
        [52:56]: Golem 5 | 0 | 0 | 0
        [56:60]: Lich 6 | 0 | 0 | 0
        [60:64]: Demon 7 | 0 | 0 | 0
        [64:68]: Dragon 9 | 0 | 0 | 0
    [68:81]: Thứ tự quái vật chưa mở
    [81]: Sô quái vật chưa mở
    [82]: Số quái vật trong hang
    [83:91]: Thông tin anh hùng Barbarian
        [83]: Được chọn trong vòng chơi 
        [84]: Máu của Barbarian
        [85]: Leather Shield: HP+3
        [86]: Healing Potion: When Barbarian dies, comeback with 4 HP (once per dungeon)
        [87]: Torch: Defeat moster with strength 3 or less
        [88]: War Hammer: Defeat Golems
        [89]: Vorpal Axe: Defeat one monster after you draw it (once per dungeon)
        [90]: Chainmail: HP+4
    [91:99]: Thông tin anh hùng Mage
        [91]: Được chọn trong vòng chơi 
        [92]: Máu của Mage
        [93]: Omnipotence: If all the monsters in the Dungeon are different, you win the round
        [94]: Holy Grail: Defeat monster with even-numbered strength
        [95]: Demonic Pact: Defeat the Demon and the next monster
        [96]: Polymorph: Defeat one monster you draw, replacing it with the next monster from the deck (once per Dungeon)
        [97]: Wall Of Fire: HP+6
        [98]: Bracelet Of Protection: HP+3
    env[99]: Turn
    env[100]: Phase (0: Bidding, 1: Dungeon)
    env[101:114]: Lá quái vật nào vừa mở (1 là mở, 0 là không)


