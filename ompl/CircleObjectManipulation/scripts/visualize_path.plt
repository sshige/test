# set term gif animate optimize delay 2 size 480,360
# set output 'movie.gif'

set xr [-2:2]
set yr [-1.5:1.5]
set size ratio -1
set parametric

file="/tmp/ompl_path.txt"
line_num=system("wc -l " . file . " | awk '{print $1}'")

set object line_num+1 circle at 0,1 size 0.75 fc rgb "green" fs solid
set object line_num+2 circle at 0,-1 size 0.75 fc rgb "green" fs solid

do for [i = 1:line_num-1 ] {
   obj_x=system("sed -n " . i . "p " . file . " | awk '{print $1}'")
   obj_y=system("sed -n " . i . "p " . file . " | awk '{print $2}'")
   set object i circle at obj_x,obj_y size 0.2 fc rgb "red" lw 2
   if(i != 1) {
        set arrow i from obj_x_prev,obj_y_prev to obj_x,obj_y fc rgb "orange" lw 3
   }
   obj_x_prev=obj_x
   obj_y_prev=obj_y
   plot 0,0
   }

set out
set terminal wxt enhanced

pause -1 "hit Enter key"
