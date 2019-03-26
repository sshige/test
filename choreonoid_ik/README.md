# choreonoid_ik

## Build
```bash
catkin build choreonoid_ik
```

## Execute
```
roscd choreonoid_ik/euslisp/sample/
roseus sample-eus-choreonoid-converter.l
(sample-eus-choreonoid-converter-sample-robot) ;; :wait? t for step execution
(sample-eus-choreonoid-converter-pa10) ;; :wait? t for step execution
```

## Misc
```bash
# FK and IK sample
rosrun choreonoid_ik sample_ik
```

Same program with euslisp:
```bash
# FK and IK sample
roscd choreonoid_ik/euslisp
roseus sample-ik.l
```
