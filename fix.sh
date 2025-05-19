#!/bin/bash
for i in {1..12}; do
  mkdir -p dataset$i/images dataset$i/labels
  mv dataset$i/obj_train_data/*.jpg dataset$i/images/ 2>/dev/null
  mv dataset$i/obj_train_data/*.png dataset$i/images/ 2>/dev/null
  mv dataset$i/obj_train_data/*.txt dataset$i/labels/ 2>/dev/null
done
