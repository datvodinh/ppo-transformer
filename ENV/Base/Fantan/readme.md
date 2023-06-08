##  :video_game: Action

      ACTION_SIZE: 53
      [0:52]: Bài muốn đánh
      [52]: bỏ lượt

##  :bust_in_silhouette: Player_state
      STATE_SIZE: 112
      [0:52]: Player card
      [52:104]: Những lá có thể đánh
            Giá trị bài: k % 13
                  0 là lá 1...
            Loại bài: k // 13
                  0: Cơ,1: rô,2: chuồn,3: bích
      [104]: Player chip
      [105:108]: Độ dài lá bài còn lại của 3 người chơi
      [108]: Game đã kết thúc hay chưa
      [109:112]: Chip của 3 người chơi còn lại
##  :globe_with_meridians: env
      
      [0:8]: 8 lá có thể đánh trên bàn
      [8:21]: Bài người chơi 1
      [22:35]: Bài người chơi 2
      [36:49]: Bài người chơi 3
      [50:63]: Bài người chơi 4
      [21,35,49,63]: chip của 4 người chơi
      [64]: Id người chơi
      [65]: Hũ chip chung
      [66]: Game đã kết thúc hay chưa

