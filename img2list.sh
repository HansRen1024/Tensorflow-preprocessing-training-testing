#!/bin/sh  
  
classes=(balcony bathroom shower bedroom bedchamber childs_room)  
num=0  
for class in ${classes[@]}  
do  
    ls $class/* > $class.txt  
    sed -i "s/$/ $num/g" $class.txt #add labels 
    let num+=1  
    cat $class.txt >> temp.txt  
    rm $class.txt  
done  
cat temp.txt | awk 'BEGIN{srand()}{print rand()"\t"$0}' | sort -k1,1 -n | cut -f2- > train.txt #shuffle  
