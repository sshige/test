(load "package://eus_qp/euslisp/eiquadprog.l")
(load "package://eus_qpoases/euslisp/eus-qpoases.l")

;; setup ;;;;;;;;;;;;;;;;;;
(defun setup-object
  ()
  (let* ((lx-max 1000.0)
         (lx-min 200.0)
         (ly-max 1000.0)
         (ly-min 200.0)
         (lz-max 1000.0)
         (lz-min 200.0)
         (lx (+ (random (- lx-max lx-min)) lx-min))
         (ly (+ (random (- ly-max ly-min)) ly-min))
         (lz (+ (random (- lz-max lz-min)) lz-min))
         )
    (setq *obj* (make-cube lx ly lz))
    (send *obj* :newcoords (get-random-coords))
    (send *obj* :set-color (float-vector 0.8 0.5 0.5) 0.5)
    (setq *model-obj* (copy-object *obj*))
    (send *model-obj* :set-color (float-vector 0.5 0.8 0.5) 0.5)
    (objects (list *obj* *model-obj*))
    (send *irtviewer* :change-background (float-vector 1.0 1.0 1.0))
    (send *irtviewer* :draw-objects)
    ))


;; solve optimization ;;;;;;;;;;;;;;;;;;
(defun refine-pose-for-orthogonal
  (pose-vec constraint-coeff)
  (let* ((t-vec
          (subseq pose-vec 0 3))
         (r-mat
          (vec2mat (subseq pose-vec 3 12)))
         (svd-ret
          (sv-decompose r-mat))
         (u-mat
          (elt svd-ret 0))
         (sigma-vec
          (elt svd-ret 1))
         (v-mat
          (elt svd-ret 2))
         (uv-det
          (matrix-determinant (m* u-mat (transpose v-mat))))
         (refined-r-mat
          (m* (m* u-mat (diagonal (float-vector 1 1 uv-det))) (transpose v-mat)))
         )
    (let* ((f-mat
            (cadr (assoc :f-mat constraint-coeff)))
           (d-vec
            (cadr (assoc :d-vec constraint-coeff)))
           (refined-r-vec
            (mat2vec refined-r-mat))
           (f-mat-t
            (block-matrix f-mat 0 0 nil 3))
           (f-mat-r
            (block-matrix f-mat 0 3 nil 9))
           (refined-t-vec
            (transform (pseudo-inverse f-mat-t) (v- (v- d-vec) (transform f-mat-r refined-r-vec))))
           )
      (list
       (list :t-vec refined-t-vec)
       (list :r-mat refined-r-mat)
       )
      )))


(defun get-pose-in-vector-format-with-qp
  (constraint-coeff
   &key
   (qp-solver #'solve-qpoases-qp)
   (debug? nil)
   )
  "input coeff is vector-pose format."
  (let* ((f-mat
          (cadr (assoc :f-mat constraint-coeff)))
         (d-vec
          (cadr (assoc :d-vec constraint-coeff)))
         ;; optimize function
         (eval-weight-matrix
          (unit-matrix 12))
         (eval-coeff-vector
          (instantiate float-vector 12))
         ;; equality
         (equality-matrix
          f-mat)
         (equality-vector
          (v- d-vec))
         ;; inequality
         (inequality-matrix
          nil)
         (inequality-min-vector
          nil)
         ;; initial
         (initial-state
          (float-vector 0 0 0 1 0 0 0 1 0 0 0 1))
         opt-value
         )
    (when debug?
      (when eval-weight-matrix
        (warning-message 2 "eval-weight-matrix:~%")
        (format-array eval-weight-matrix))
      (when eval-coeff-vector
        (warning-message 2 "eval-coeff-vector:~%")
        (format-array eval-coeff-vector))
      (when equality-matrix
        (warning-message 2 "equality-matrix:~%")
        (format-array equality-matrix))
      (when equality-vector
        (warning-message 2 "equality-vector:~%")
        (format-array equality-vector))
      (when inequality-matrix
        (warning-message 2 "inequality-matrix:~%")
        (format-array inequality-matrix))
      (when inequality-min-vector
        (warning-message 2 "inequality-min-vector:~%")
        (format-array inequality-min-vector))
      )
    (let* ((pose-vec
            (apply qp-solver
                   :initial-state initial-state
                   :eval-weight-matrix eval-weight-matrix
                   :eval-coeff-vector eval-coeff-vector
                   :equality-matrix equality-matrix
                   :equality-vector equality-vector
                   :inequality-matrix inequality-matrix
                   :inequality-min-vector inequality-min-vector
                   :debug nil
                   (if (equal (car (send qp-solver :cdr)) 'solve-eiquadprog)
                       nil (list :check-status t))
                   ))
           )
      (unless pose-vec
        (warning-message 1 "[qp] qp failed.~%"))
      pose-vec
      )))


;; constraint coefficient ;;;;;;;;;;;;;;;;;;
(defun get-constraint-coeff-from-vision-info
  (vision-info
   &key
   (debug? nil)
   )
  "return coeff in vector-pose format."
  (let* ((edge-list (elt vision-info 0))
         (face-list (elt vision-info 1))
         (a-vec-list)
         (b-vec-list)
         (c-vec-list)
         (d-list)
         (f-mat)
         (d-vec)
         )
    (dolist (edge0 edge-list)
      (let* ((coeff-list (get-constraint-coeff-from-edge edge0))
             )
        (setq a-vec-list (concatenate cons (cadr (assoc :a-vec-list coeff-list)) a-vec-list))
        (setq b-vec-list (concatenate cons (cadr (assoc :b-vec-list coeff-list)) b-vec-list))
        (setq c-vec-list (concatenate cons (cadr (assoc :c-vec-list coeff-list)) c-vec-list))
        (setq d-list (concatenate cons (cadr (assoc :d-list coeff-list)) d-list))
        ))
    (dolist (face0 face-list)
      (let* ((coeff-list (get-constraint-coeff-from-face face0))
             )
        (setq a-vec-list (concatenate cons (cadr (assoc :a-vec-list coeff-list)) a-vec-list))
        (setq b-vec-list (concatenate cons (cadr (assoc :b-vec-list coeff-list)) b-vec-list))
        (setq c-vec-list (concatenate cons (cadr (assoc :c-vec-list coeff-list)) c-vec-list))
        (setq d-list (concatenate cons (cadr (assoc :d-list coeff-list)) d-list))
        ))
    (when debug?
      (pprint
       (list
        (list :a-vec-list a-vec-list)
        (list :b-vec-list b-vec-list)
        (list :c-vec-list c-vec-list)
        (list :d-list d-list)
        )))
    (setq d-vec (coerce d-list float-vector))
    (dotimes (i (length a-vec-list))
      (cond (f-mat
             (setq f-mat (concatenate-matrix-column f-mat (convert-constraint-coeff-from-se3-to-vector
                                                           (elt a-vec-list i) (elt b-vec-list i) (elt c-vec-list i))))
             )
            (t
             (setq f-mat (convert-constraint-coeff-from-se3-to-vector
                          (elt a-vec-list i) (elt b-vec-list i) (elt c-vec-list i)))
             ))
      )
    (list
     (list :d-vec d-vec)
     (list :f-mat f-mat))
    ))

(defun get-constraint-coeff-from-edge
  (edge-param-id)
  "return coeff in SE3-pose format."
  (let* ((edge-param (elt edge-param-id 0))
         (edge-pos (elt edge-param 0))
         (edge-dir (elt edge-param 1))
         (edge-id (elt edge-param-id 1))
         (obj-edge-param-local (get-edge-param (elt (send *obj* :edges) edge-id) :obj-frame? t))
         (obj-edge-pos-local (elt obj-edge-param-local 0))
         (obj-edge-dir-local (elt obj-edge-param-local 1))
         (lambda-xy (v- (scale (elt edge-dir 1) (float-vector 1 0 0)) (scale (elt edge-dir 0) (float-vector 0 1 0))))
         (lambda-xz (v- (scale (elt edge-dir 2) (float-vector 1 0 0)) (scale (elt edge-dir 0) (float-vector 0 0 1))))
         (a-vec-list)
         (b-vec-list)
         (c-vec-list)
         (d-list)
         )
    ;; constraint 1
    (push lambda-xy a-vec-list)
    (push obj-edge-dir-local b-vec-list)
    (push (float-vector 0 0 0) c-vec-list)
    (push 0 d-list)
    ;; constraint 2
    (push lambda-xz a-vec-list)
    (push obj-edge-dir-local b-vec-list)
    (push (float-vector 0 0 0) c-vec-list)
    (push 0 d-list)
    ;; constraint 3
    (push lambda-xy a-vec-list)
    (push obj-edge-pos-local b-vec-list)
    (push lambda-xy c-vec-list)
    (push (- (v. lambda-xy edge-pos)) d-list)
    ;; constraint 4
    (push lambda-xz a-vec-list)
    (push obj-edge-pos-local b-vec-list)
    (push lambda-xz c-vec-list)
    (push (- (v. lambda-xz edge-pos)) d-list)
    (list
     (list :a-vec-list a-vec-list)
     (list :b-vec-list b-vec-list)
     (list :c-vec-list c-vec-list)
     (list :d-list d-list)
     )
    ))

(defun get-constraint-coeff-from-face
  (face-param-id)
  "return coeff in SE3-pose format."
  (let* ((face-param (elt face-param-id 0))
         (face-pos (elt face-param 0))
         (face-norm (elt face-param 1))
         (face-id (elt face-param-id 1))
         (obj-face-param-local (get-face-param (elt (send *obj* :faces) face-id) :obj-frame? t))
         (obj-face-pos-local (elt obj-face-param-local 0))
         (obj-face-norm-local (elt obj-face-param-local 1))
         (a-vec-list)
         (b-vec-list)
         (c-vec-list)
         (d-list)
         )
    ;; constraint 1
    (push (v- (scale (elt face-norm 1) (float-vector 1 0 0)) (scale (elt face-norm 0) (float-vector 0 1 0))) a-vec-list)
    (push obj-face-norm-local b-vec-list)
    (push (float-vector 0 0 0) c-vec-list)
    (push 0 d-list)
    ;; constraint 2
    (push (v- (scale (elt face-norm 2) (float-vector 1 0 0)) (scale (elt face-norm 0) (float-vector 0 0 1))) a-vec-list)
    (push obj-face-norm-local b-vec-list)
    (push (float-vector 0 0 0) c-vec-list)
    (push 0 d-list)
    ;; constraint 3
    (push face-norm a-vec-list)
    (push obj-face-pos-local b-vec-list)
    (push face-norm c-vec-list)
    (push (- (v. face-norm face-pos)) d-list)
    (list
     (list :a-vec-list a-vec-list)
     (list :b-vec-list b-vec-list)
     (list :c-vec-list c-vec-list)
     (list :d-list d-list)
     )
    ))

(defun convert-constraint-coeff-from-se3-to-vector
  (a-vec b-vec c-vec)
  (assert (= (length a-vec) 3))
  (assert (= (length b-vec) 3))
  (assert (= (length c-vec) 3))
  (let* ((a-mat (matrix a-vec))
         (b-mat (matrix b-vec))
         (c-mat (matrix c-vec))
         )
    (concatenate-matrix-row c-mat (scale-matrix (aref b-mat 0 0) a-mat) (scale-matrix (aref b-mat 0 1) a-mat) (scale-matrix (aref b-mat 0 2) a-mat))
    ))


;; sensor ;;;;;;;;;;;;;;;;;;
(defun get-random-vision-info
  ()
  (let* ((edge-num 0)
         (face-num 2)
         (edge-list)
         (face-list)
         )
    (dotimes (i edge-num)
      (let* ((edge-id
              (get-random-edge))
             (edge-param-id
              (list (get-edge-param (elt edge-id 0) :noise? nil) (elt edge-id 1)))
             )
        (push edge-param-id edge-list)
        ))
    (dotimes (i face-num)
      (let* ((face-id
              (get-random-face))
             (face-param-id
              (list (get-face-param (elt face-id 0) :noise? nil) (elt face-id 1)))
             )
        (push face-param-id face-list)
        ))
    (list edge-list face-list)
    ))


;; object ;;;;;;;;;;;;;;;;;;
;; edge
(defun draw-edge-param
  (edge-param)
  (let* ((pos (elt edge-param 0))
         (dir (elt edge-param 1))
         )
    (with-set-color-line-width
     (send *irtviewer* :viewer) #F(1 0 0) 10
     (send *irtviewer* :viewer :draw-arrow
           pos (v+ pos (scale 200 dir))))
    ))

(defun get-edge-param
  (edge0
   &key
   (noise? nil)
   (obj-frame? nil)
   )
  (let* ((pos (send edge0 :point 0.5))
         (dir (send edge0 :direction))
         (pos-noise-norm 10.0)
         (dir-noise-norm 0.1)
         )
    (when obj-frame?
      (setq pos (send (send *obj* :worldcoords) :inverse-transform-vector pos))
      (setq dir (send (send *obj* :worldcoords) :inverse-rotate-vector dir))
      )
    (when noise?
      (setq pos (v+ pos (random-vector pos-noise-norm)))
      (setq dir (v+ dir (random-vector dir-noise-norm)))
      )
    (list pos dir)
    ))

(defun get-random-edge
  ()
  (let* ((id0 (random (length (send *obj* :edges))))
         )
    (list (elt (send *obj* :edges) id0) id0)
    ))

;; face
(defun draw-face-param
  (face-param)
  (let* ((pos (elt face-param 0))
         (norm (elt face-param 1))
         )
    (with-set-color-line-width
     (send *irtviewer* :viewer) #F(1 0 0) 10
     (send *irtviewer* :viewer :draw-arrow
           pos (v+ pos (scale 200 norm))))
    ))

(defun get-face-param
  (face0
   &key
   (noise? nil)
   (obj-frame? nil)
   )
  (let* ((pos (cadr (send face0 :centroid)))
         (norm (send face0 :normal))
         (pos-noise-norm 10.0)
         (norm-noise-norm 0.1)
         )
    (when obj-frame?
      (setq pos (send (send *obj* :worldcoords) :inverse-transform-vector pos))
      (setq norm (send (send *obj* :worldcoords) :inverse-rotate-vector norm))
      )
    (when noise?
      (setq pos (v+ pos (random-vector pos-noise-norm)))
      (setq norm (v+ norm (random-vector norm-noise-norm)))
      )
    (list pos norm)
    ))

(defun get-random-face
  ()
  (let* ((id0 (random (length (send *obj* :faces))))
         )
    (list (elt (send *obj* :faces) id0) id0)
    ))


;; math ;;;;;;;;;;;;;;;;;;
(defun vec2mat
  (vec)
  (let* ((r1-vec
          (subseq vec 0 3))
         (r2-vec
          (subseq vec 3 6))
         (r3-vec
          (subseq vec 6 9))
         )
    (transpose (matrix r1-vec r2-vec r3-vec))
    ))

(defun mat2vec
  (mat)
  (concatenate float-vector (matrix-column mat 0) (matrix-column mat 1) (matrix-column mat 2))
  )

(defun block-matrix
  (m &optional (i 0) (j 0) (ii nil) (jj nil))
  ;; i, j : start row and column idx
  ;; ii, jj : row and column length of block matrix
  (unless ii (setq ii (- (array-dimension m 0) i)))
  (unless jj (setq jj (- (array-dimension m 1) j)))
  (let ((ret (make-matrix ii jj)))
    (dotimes (_i ii)
      (dotimes (_j jj)
        (setf (aref ret _i _j) (aref m (+ _i i) (+ _j j)))))
    ret))

(defun block-vec
  ;; i : start idx
  ;; ii : length of block vector
  (m &optional (i 0) (ii nil))
  (unless ii (setq ii (- (length m) i)))
  (let ((ret (instantiate float-vector ii)))
    (dotimes (_i ii)
      (setf (aref ret _i) (aref m (+ _i i))))
    ret))


;; utils ;;;;;;;;;;;;;;;;;;
(defun get-random-coords
  ()
  (let* ((pos (random-vector (random 1000.0)))
         (roll-min (- pi))
         (roll-max pi)
         (pitch-min (- pi))
         (pitch-max pi)
         (yaw-min (- pi))
         (yaw-max pi)
         (roll (+ (random (- roll-max roll-min)) roll-min))
         (pitch (+ (random (- pitch-max pitch-min)) pitch-min))
         (yaw (+ (random (- yaw-max yaw-min)) yaw-min))
         (rpy (list yaw pitch roll))
         )
    (make-coords :pos pos :rpy rpy)
    ))

(defmacro with-set-color-line-width
  (vw color line-width &rest bodies)
  (with-gensyms
   (prev-color prev-width)
   `(let* ((,prev-color (send ,vw :viewsurface :color))
           (,prev-width (send ,vw :viewsurface :line-width)))
      (unwind-protect
          (progn
            (send ,vw :viewsurface :color ,color)
            (send ,vw :viewsurface :line-width ,line-width)
            ,@bodies)
        (send ,vw :viewsurface :color ,prev-color)
        (send ,vw :viewsurface :line-width ,prev-width)
        ))))


;; test ;;;;;;;;;;;;;;;;;;
(defun test
  ()
  (setup-object)
  (let* ((vision-info
          (get-random-vision-info))
         (edge-list
          (elt vision-info 0))
         (face-list
          (elt vision-info 1))
         (constraint-coeff
          (get-constraint-coeff-from-vision-info vision-info :debug? nil))
         (pose-vec
          (get-pose-in-vector-format-with-qp constraint-coeff :debug? nil))
         (est-pose
          (refine-pose-for-orthogonal pose-vec constraint-coeff))
         (est-pos (cadr (assoc :t-vec est-pose)))
         (est-rot (cadr (assoc :r-mat est-pose)))
         (est-rpy (coerce (car (rpy-angle est-rot)) float-vector))
         (gt-pos (send *obj* :worldpos))
         (gt-rpy (coerce (car (rpy-angle (send *obj* :worldrot))) float-vector))
         )

    (send *model-obj* :newcoords (make-coords :pos est-pos :rot est-rot))
    (send *irtviewer* :draw-objects)

    (dolist (edge0 edge-list)
      (let* ((edge-param (elt edge0 0))
             (edge-id (elt edge0 1))
             )
        (send (elt (send *obj* :edges) edge-id) :draw-on :flush t :color #f(1 0 0) :width 5)
        (send (elt (send *model-obj* :edges) edge-id) :draw-on :flush t :color #f(0 1 0) :width 5)
        ))
    (dolist (face0 face-list)
      (let* ((face-param (elt face0 0))
             (face-id (elt face0 1))
             )
        (send (elt (send *obj* :faces) face-id) :draw-on :flush t :color #f(1 0 0) :width 5)
        (send (elt (send *model-obj* :faces) face-id) :draw-on :flush t :color #f(0 1 0) :width 5)
        ))

    (format-array (cadr (assoc :f-mat constraint-coeff)) "f-mat: ")
    (format-array (cadr (assoc :d-vec constraint-coeff)) "d-vec: ")

    (warning-message 2 "pos~%")
    (format-array est-pos "estimated:    ")
    (format-array gt-pos "ground truth: ")
    (warning-message 2 "rpy~%")
    (format-array est-rpy "estimated:    ")
    (format-array gt-rpy "ground truth: ")

    (let* ((f-mat (cadr (assoc :f-mat constraint-coeff)))
           (d-vec (cadr (assoc :d-vec constraint-coeff)))
           (refined-pose-vec (concatenate float-vector est-pos (mat2vec est-rot)))
           )
      (warning-message 1 "constraint error without refine: ~a~%" (v+ (transform f-mat pose-vec) d-vec))
      (warning-message 1 "constraint error with refine: ~a~%" (v+ (transform f-mat refined-pose-vec) d-vec))
      )
    ))

(defun test-tmp
  ()
  (send *obj* :newcoords (make-coords :rpy (list 0.3 -0.3 -0.3)))
  (let* ((edge-id 7)
         (vision-info
          (list (list (list (get-edge-param (elt (send *obj* :edges) edge-id)) edge-id)) nil)
          (get-random-vision-info))
         (edge-list
          (elt vision-info 0))
         (face-list
          (elt vision-info 1))
         (constraint-coeff
          (get-constraint-coeff-from-vision-info vision-info :debug? nil))
         (pose-vec
          (get-pose-in-vector-format-with-qp constraint-coeff :debug? nil))
         (est-pose
          (refine-pose-for-orthogonal pose-vec constraint-coeff))
         (est-pos (cadr (assoc :t-vec est-pose)))
         (est-rot (cadr (assoc :r-mat est-pose)))
         (est-rpy (coerce (car (rpy-angle est-rot)) float-vector))
         (gt-pos (send *obj* :worldpos))
         (gt-rpy (coerce (car (rpy-angle (send *obj* :worldrot))) float-vector))
         )

    (dolist (edge0 edge-list)
      (let* ((edge-param (elt edge0 0))
             (edge-id (elt edge0 1))
             )
        (send (elt (send *obj* :edges) edge-id) :draw-on :flush t :color #f(1 0 0) :width 5)
        ))
    (dolist (face0 face-list)
      (let* ((face-param (elt face0 0))
             (face-id (elt face0 1))
             )
        (send (elt (send *obj* :faces) face-id) :draw-on :flush t :color #f(1 0 0) :width 5)
        ))
    (send *model-obj* :newcoords (make-coords :pos est-pos :rot est-rot))
    (send *model-obj* :worldcoords)
    (send *model-obj* :draw-on :flush t :color #f(0 1 0) :width 5)

    (format-array (cadr (assoc :f-mat constraint-coeff)) "f-mat: ")
    (format-array (cadr (assoc :d-vec constraint-coeff)) "d-vec: ")

    (warning-message 1 "pos~%")
    (format-array est-pos "estimated:    ")
    (format-array gt-pos "ground truth: ")
    (warning-message 1 "rpy~%")
    (format-array est-rpy "estimated:    ")
    (format-array gt-rpy "ground truth: ")

    (let* ((f-mat (cadr (assoc :f-mat constraint-coeff)))
           (d-vec (cadr (assoc :d-vec constraint-coeff)))
           (refined-pose-vec (concatenate float-vector est-pos (mat2vec est-rot)))
           )
      (warning-message 1 "constraint error without refine: ~a~%" (v+ (transform f-mat pose-vec) d-vec))
      (warning-message 1 "constraint error with refine: ~a~%" (v+ (transform f-mat refined-pose-vec) d-vec))
      )

    ))



#|
(format-array (cadr (assoc :f-mat (get-constraint-coeff-from-vision-info (get-random-vision-info)))))
(get-pose-in-vector-format-with-qp (get-constraint-coeff-from-vision-info (get-random-vision-info)) :debug? t)
(refine-pose-for-orthogonal (get-pose-in-vector-format-with-qp (get-constraint-coeff-from-vision-info (get-random-vision-info)) :debug? nil))
