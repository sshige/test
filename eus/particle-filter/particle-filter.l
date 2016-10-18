;; init ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun init
  ()
  (setq *obj* (make-cube 10 20 30))
  (setq *tmp-obj* (copy-object *obj*))
  (send *obj* :newcoords (make-coords))
  (objects (list *obj*))

  (init-particle)
  )

(defun init-particle
  (&key (particle-num 1000))
  (setq *particle-list* nil)
  (dotimes (i particle-num)
    (push (make-coords) *particle-list*)
    )
  )

;; motion model ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun motion-model
  (coords)
  (send (send coords :copy-worldcoords) :translate (float-vector 2 0 0) :local)
  )

(defun motion-model-with-error
  (coords
   &key
   (trans-sd 5.0)
   )
  (send (motion-model coords) :translate (gaussian-random 3 0 trans-sd) :local)
  )

(defun apply-motion-model-with-error
  ()
  (send *obj* :newcoords (motion-model-with-error (send *obj* :worldcoords)))
  )

;; sensor model ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun sensor-model
  (coords)
  (let* (intersection-bbox
         )
    (send *tmp-obj* :newcoords coords)
    (send *tmp-obj* :worldcoords) ;; for updating
    (setq intersection-bbox (send (send *obj* :box) :intersection (send *tmp-obj* :box)))
    (if intersection-bbox (send intersection-bbox :volume) *epsilon*)
    ))

;; filter ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun filter
  ()
  (let* ((weight-sum 0)
         (new-particle-list nil)
         )
    ;; apply motion model and calculate weight
    (dotimes (i (length *particle-list*))
      (setf (elt *particle-list* i) (motion-model-with-error (elt *particle-list* i)))
      (send (elt *particle-list* i) :put :weight (sensor-model (elt *particle-list* i)))
      (setq weight-sum (+ weight-sum (send (elt *particle-list* i) :get :weight)))
      )
    ;; normalize weight
    (dotimes (i (length *particle-list*))
      (send (elt *particle-list* i) :put :weight (/ (send (elt *particle-list* i) :get :weight) weight-sum))
      )
    ;; resample (Probabilistic Robotics p.101)
    (let* ((M-inv (/ 1.0 (length *particle-list*)))
           (r (random M-inv))
           (c (send (elt *particle-list* 0) :get :weight))
           (u r)
           (i 0)
           )
      (dotimes (k (length *particle-list*))
        (unless (= k 0)
          (setq u (+ u M-inv)))
        (while (> u c)
          (setq i (min (1+ i) (1- (length *particle-list*))))
          (setq c (+ c (send (elt *particle-list* i) :get :weight)))
          )
        (push (elt *particle-list* i) new-particle-list)
        ))
    (setq *particle-list* new-particle-list)

    nil))

;; util ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun animate-motion-model
  ()
  (do-until-key
   (apply-motion-model-with-error)
   (print (sensor-model (send (send *obj* :worldcoords) :copy-worldcoords)))
   (unix::usleep (* 100 1000))
   (send *irtviewer* :draw-objects)
   )
  )

(defun visualize-particle
  ()
  (dolist (particle *particle-list*)
    (visualize-particle-one particle)
    )
  (send *irtviewer* :viewer :flush)
  )

(defun visualize-particle-one
  (coords)
  (send *tmp-obj* :newcoords coords)
  (send *tmp-obj* :worldcoords) ;; for updating
  (send *tmp-obj* :draw-on :color (float-vector 1 0 0))
  )

;; main ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defun main
  ()
  (init)
  (do-until-key
   (apply-motion-model-with-error)
   (filter)
   (x::window-main-one)
   (send *irtviewer* :draw-objects :flush nil)
   (visualize-particle)
   (unix::usleep (* 10 1000))
   )
  )
