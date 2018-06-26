# set term gif animate optimize delay 2 size 480,360
set output 'movie2.gif'

set xr [-2:2]
set yr [-1.5:1.5]
set size ratio -1
set parametric

file="/tmp/ompl_path.txt"
line_num=system("wc -l " . file . " | awk '{print $1}'")

rad2deg(x)=x*180.0/pi

set object 10000 circle at 0.5,0.0 size 0.5 fc rgb "blue" lw 1

do for [i = 1:line_num-1 ] {
   theta_0=system("sed -n " . i . "p " . file . " | awk '{print $1}'")
   theta_1=system("sed -n " . i . "p " . file . " | awk '{print $2}'")

   # whether to draw trajectory or not
   draw_trajectory=0

   robot_width=2

   if (!draw_trajectory) {
       i=1
   }

   # calc FK
   j0_x=0.0
   j0_y=0.0
   j1_x=cos(theta_0)
   j1_y=sin(theta_0)
   j2_x=j1_x+cos(theta_0+theta_1)
   j2_y=j1_y+sin(theta_0+theta_1)

   # draw object
   set arrow 2*i from j0_x,j0_y to j1_x,j1_y fc rgb "red" lw 10 nohead
   set arrow 2*i+1 from j1_x,j1_y to j2_x,j2_y fc rgb "red" lw 10 nohead

   plot 0,0 notitle
   pause 0.05
   }

set out
set terminal wxt enhanced

pause -1 "hit Enter key"
