# set term gif animate optimize delay 2 size 480,360
# set output 'movie2.gif'

set xr [-2:2]
set yr [-1.5:1.5]
set size ratio -1
set parametric

file="/tmp/ompl_path.txt"
line_num=system("wc -l " . file . " | awk '{print $1}'")

# draw environment
set object line_num*10+1 circle at 0,1 size 0.75 fc rgb "green" fs solid
set object line_num*10+2 circle at 0,-1 size 0.75 fc rgb "green" fs solid

rad2deg(x)=x*180.0/pi

do for [i = 1:line_num ] {
   obj_x=system("sed -n " . i . "p " . file . " | awk '{print $1}'")
   obj_y=system("sed -n " . i . "p " . file . " | awk '{print $2}'")
   obj_theta=system("sed -n " . i . "p " . file . " | awk '{print $3}'")
   contact1=system("sed -n " . i . "p " . file . " | awk '{print $4}'")
   contact2=system("sed -n " . i . "p " . file . " | awk '{print $5}'")
   contact3=system("sed -n " . i . "p " . file . " | awk '{print $6}'")
   contact4=system("sed -n " . i . "p " . file . " | awk '{print $7}'")

   # whether to draw trajectory or not
   draw_trajectory=0

   obj_radius=0.2
   gripper_radius=0.1

   if (!draw_trajectory) {
       i=1
   }

   # draw gripper
   if (i != 1) {
        set arrow i from obj_x_prev,obj_y_prev to obj_x,obj_y fc rgb "orange" lw 3
   }
   if (contact1 == 1) {
      gripper_x=(gripper_radius + obj_radius) * cos(obj_theta) + obj_x
      gripper_y=(gripper_radius + obj_radius) * sin(obj_theta) + obj_y
      set object i*6+2 circle at gripper_x,gripper_y size gripper_radius fc rgb "#9932CC" lw 2
   } else {
      if (!draw_trajectory) {
         unset object i*6+2
      }
   }
   if (contact2 == 1) {
      gripper_x=(gripper_radius + obj_radius) * cos(obj_theta+pi*0.5) + obj_x
      gripper_y=(gripper_radius + obj_radius) * sin(obj_theta+pi*0.5) + obj_y
      set object i*6+3 circle at gripper_x,gripper_y size gripper_radius fc rgb "#008B8B" lw 2
   } else {
      if (!draw_trajectory) {
         unset object i*6+3
      }
   }
   if (contact3 == 1) {
      gripper_x=(gripper_radius + obj_radius) * cos(obj_theta+pi) + obj_x
      gripper_y=(gripper_radius + obj_radius) * sin(obj_theta+pi) + obj_y
      set object i*6+4 circle at gripper_x,gripper_y size gripper_radius fc rgb "#0000CD" lw 2
   } else {
      if (!draw_trajectory) {
         unset object i*6+4
      }
   }
   if (contact4 == 1) {
      gripper_x=(gripper_radius + obj_radius) * cos(obj_theta+pi*1.5) + obj_x
      gripper_y=(gripper_radius + obj_radius) * sin(obj_theta+pi*1.5) + obj_y
      set object i*6+5 circle at gripper_x,gripper_y size gripper_radius fc rgb "#4B0082" lw 2
   } else {
      if (!draw_trajectory) {
         unset object i*6+5
      }
   }

   # draw object
   set object i*6 circle at obj_x,obj_y size obj_radius fc rgb "red" lw 2
   set object i*6+1 circle arc [-10+rad2deg(obj_theta):10+rad2deg(obj_theta)] at obj_x,obj_y size 0.2 fc rgb "red" fs solid lw 1

   obj_x_prev=obj_x
   obj_y_prev=obj_y
   plot 0,0 notitle
   }

set out
set terminal wxt enhanced

pause -1 "hit Enter key"
