(setq *obj* (make-cone (float-vector 0 0 100) (reverse (list (float-vector -50 -50 0) (float-vector 50 -50 0) (float-vector 50 50 0) (float-vector -50 50 0)))))
(objects (list *obj*))

(send *obj* :name "grasp_cone")
(collada::eus2collada *obj* "/tmp")
;; rosrun collada_urdf collada_to_urdf "/tmp/grasp_cone.dae" -G -A --mesh_output_dir meshes --mesh_prefix "model://%s/meshes" -O %s
