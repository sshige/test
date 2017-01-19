(load "package://eus_qp/euslisp/eiquadprog.l")
(load "package://eus_qpoases/euslisp/eus-qpoases.l")
(load "package://eus_nlopt/euslisp/nlopt.l")

(load "package://wholebody_manipulation_planner/euslisp/approx-obj-poly-body.l")


(defvar *ex* (float-vector 1 0 0))
(defvar *ey* (float-vector 0 1 0))
(defvar *ez* (float-vector 0 0 1))

(defvar *-ex* (float-vector -1 0 0))
(defvar *-ey* (float-vector 0 -1 0))
(defvar *-ez* (float-vector 0 0 -1))

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
    ;; (setq *obj* (approx-blue-cart-poly-body))
    ;; (setq *obj* (approx-drum-can-poly-body))
    (send *obj* :set-color (float-vector 0.8 0.5 0.5) 0.5)

    ;; assoc grasp point
    (let* ((grasp-pos
            (float-vector (- (random lx) (* 0.5 lx)) (- (random ly) (* 0.5 ly)) (- (random lz) (* 0.5 lz))))
           )
      (send *obj* :put :grasp-point (make-cascoords :pos grasp-pos :parent *obj*))
      (send *obj* :put :grasp-pos-in-obj-frame grasp-pos)
      )

    (send *obj* :newcoords (get-random-coords))
    (send *obj* :worldcoords)
    (send *obj* :translate (float-vector 0 0 (- (apply #'min (mapcar #'(lambda (x) (elt x 2)) (send *obj* :vertices))))) :world)

    (setq *model-obj* (copy-object *obj*))
    (send *model-obj* :set-color (float-vector 0.5 0.8 0.5) 0.5)

    (objects (list *obj* *model-obj*))
    (send *irtviewer* :change-background (float-vector 0.8 0.8 0.8))
    (send *irtviewer* :set-val 'x::draw-floor 1000)
    (send *irtviewer* :draw-objects)
    ))


;; solve nonlinear optimization ;;;;;;;;;;;;;;;;;;
(defun set-vec
  (v vec)
  (dotimes (i (length vec))
    (setf (aref v i) (aref vec i))
    )
  )

(defun set-mat
  (v mat)
  (let* ((height (cdr (assoc 'dim0 (send mat :slots))))
         (width (cdr (assoc 'dim1 (send mat :slots))))
         )
    (dotimes (i height)
      (dotimes (j width)
        (setf (aref v (+ (* j height) i)) (aref mat i j))
        ))
    ))

(defun-c-callable sample-f ((v1 :string) (v2 :string)) :integer
  (setf (aref v2 0) (v. v1 v1))
  0)

(defun-c-callable sample-df ((v1 :string) (v2 :string)) :integer
  (set-vec v2 (scale 2.0 v2))
  0)

(defun-c-callable sample-g ((v1 :string) (v2 :string)) :integer
  (set-vec v2 (v- (transform *equality-matrix* v1) *equality-vector*))
  0)

(defun-c-callable sample-dg ((v1 :string) (v2 :string)) :integer
  (set-mat v2 (transpose *equality-matrix*))
  0)

(defun-c-callable sample-h ((v1 :string) (v2 :string)) :integer
  (setf (aref v2 0) 0)
  0)

(defun-c-callable sample-dh ((v1 :string) (v2 :string)) :integer
  (set-vec v2 (instantiate float-vector 12))
  0)

(defun calc-pose-vec-with-nlopt
  (constraint-coeff
   &key
   (initial-state nil)
   (qp-solver #'solve-qpoases-qp)
   (debug? nil)
   )
  "input coeff is vector-pose format."
  (let* ((f-mat-eq
          (cadr (assoc :f-mat-eq constraint-coeff)))
         (d-vec-eq
          (cadr (assoc :d-vec-eq constraint-coeff)))
         (f-mat-ineq
          (cadr (assoc :f-mat-ineq constraint-coeff)))
         (d-vec-ineq
          (cadr (assoc :d-vec-ineq constraint-coeff)))
         ;; optimize function
         (eval-weight-matrix
          (unit-matrix 12)
          )
         (eval-coeff-vector
          (instantiate float-vector 12))
         ;; equality
         (equality-matrix
          f-mat-eq)
         (equality-vector
          (if d-vec-eq (v- d-vec-eq)))
         ;; inequality
         (inequality-matrix
          f-mat-ineq)
         (inequality-min-vector
          (if d-vec-ineq (v- d-vec-ineq)))
         ;; initial
         (initial-state
          (if initial-state initial-state (float-vector 0 0 0 1 0 0 0 1 0 0 0 1)))
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

    (setq *eval-weight-matrix* eval-weight-matrix)
    (setq *eval-coeff-vector* eval-coeff-vector)
    (setq *equality-matrix* equality-matrix)
    (setq *equality-vector* equality-vector)
    (setq *inequality-matrix* inequality-matrix)
    (setq *inequality-min-vector* inequality-min-vector)

    (let* ((pose-vec
            (nlopt-optimize :ftol 1e-10 :xtol 1e-10 :eqthre 1e-10
                            :alg SLSQP

                            :initial-state initial-state
                            :state-min-vector (float-vector -10e3 -10e3 -10e3 -1 -1 -1 -1 -1 -1 -1 -1 -1)
                            :state-max-vector (float-vector 10e3 10e3 10e3 1 1 1 1 1 1 1 1 1)
                            :state-dimension 12

                            :evaluation-function 'sample-f
                            :evaluation-function-gradient 'sample-df
                            :equality-function 'sample-g
                            :equality-function-gradient 'sample-dg
                            :inequality-function 'sample-h
                            :inequality-function-gradient 'sample-dh
                            :equality-dimension (cdr (assoc 'dim0 (send *equality-matrix* :slots)))
                            :inequality-dimension 1
                            ))
           )
      (unless pose-vec
        (warning-message 1 "[qp] qp failed.~%"))
      pose-vec
      )))


;; solve QP optimization ;;;;;;;;;;;;;;;;;;
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
         (refined-r-vec
          (mat2vec refined-r-mat))
         (f-mat-eq
          (cadr (assoc :f-mat-eq constraint-coeff)))
         (d-vec-eq
          (cadr (assoc :d-vec-eq constraint-coeff)))
         (refined-t-vec t-vec)
         )
    (when f-mat-eq
      (let* ((f-mat-eq-t
              (block-matrix f-mat-eq 0 0 nil 3))
             (f-mat-eq-r
              (block-matrix f-mat-eq 0 3 nil 9))
             )
        (setq refined-t-vec
              (transform (pseudo-inverse f-mat-eq-t) (v- (v- d-vec-eq) (transform f-mat-eq-r refined-r-vec))))
        ))
    (print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    (format t "uv-det: ~a~%" uv-det)
    (format-array r-mat "R before: ")
    (format-array refined-r-mat "R after: ")
    (format-array t-vec "t before: ")
    (format-array refined-t-vec "t after: ")
    (print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    (list
     ;; (list :t-vec t-vec)
     (list :t-vec refined-t-vec)
     (list :r-mat refined-r-mat)
     )
    ))

(defun calc-pose-vec-with-qp
  (constraint-coeff
   &key
   (initial-state nil)
   (qp-solver #'solve-qpoases-qp)
   (debug? nil)
   )
  "input coeff is vector-pose format."
  (let* ((f-mat-eq
          (cadr (assoc :f-mat-eq constraint-coeff)))
         (d-vec-eq
          (cadr (assoc :d-vec-eq constraint-coeff)))
         (f-mat-ineq
          (cadr (assoc :f-mat-ineq constraint-coeff)))
         (d-vec-ineq
          (cadr (assoc :d-vec-ineq constraint-coeff)))
         ;; optimize function
         (eval-weight-matrix
          (unit-matrix 12)
          )
         (eval-coeff-vector
          (instantiate float-vector 12))
         ;; equality
         (equality-matrix
          f-mat-eq)
         (equality-vector
          (if d-vec-eq (v- d-vec-eq)))
         ;; inequality
         (inequality-matrix
          f-mat-ineq)
         (inequality-min-vector
          (if d-vec-ineq (v- d-vec-ineq)))
         ;; initial
         (initial-state
          (if initial-state initial-state (float-vector 0 0 0 1 0 0 0 1 0 0 0 1)))
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
                   :debug t
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
         (f-mat-eq)
         (d-vec-eq)
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
    (setq d-vec-eq (coerce d-list float-vector))
    (setq f-mat-eq (convert-constraint-coeff-list-from-se3-to-vector a-vec-list b-vec-list c-vec-list))
    (list
     (list :d-vec-eq d-vec-eq)
     (list :f-mat-eq f-mat-eq))
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
         (lambda-xy (v- (scale (elt edge-dir 1) *ex*) (scale (elt edge-dir 0) *ey*)))
         (lambda-xz (v- (scale (elt edge-dir 2) *ex*) (scale (elt edge-dir 0) *ez*)))
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
    (push (v- (scale (elt face-norm 1) *ex*) (scale (elt face-norm 0) *ey*)) a-vec-list)
    (push obj-face-norm-local b-vec-list)
    (push (float-vector 0 0 0) c-vec-list)
    (push 0 d-list)
    ;; constraint 2
    (push (v- (scale (elt face-norm 2) *ex*) (scale (elt face-norm 0) *ez*)) a-vec-list)
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

(defun get-constraint-coeff-from-grasp-info
  (grasp-info
   &key
   (debug? nil)
   )
  "return coeff in vector-pose format."
  (let* ((p-ro (send *obj* :get :grasp-pos-in-obj-frame))
         (p-grasp grasp-info)
         (a-vec-list
          (list *ex* *ey* *ez*))
         (b-vec-list
          (list p-ro p-ro p-ro))
         (c-vec-list
          (list *ex* *ey* *ez*))
         (d-list
          (list (- (elt p-grasp 0)) (- (elt p-grasp 1)) (- (elt p-grasp 2))))
         (d-vec-eq)
         (f-mat-eq)
         )
    (when debug?
      (pprint
       (list
        (list :a-vec-list a-vec-list)
        (list :b-vec-list b-vec-list)
        (list :c-vec-list c-vec-list)
        (list :d-list d-list)
        )))
    (setq d-vec-eq (coerce d-list float-vector))
    (setq f-mat-eq (convert-constraint-coeff-list-from-se3-to-vector a-vec-list b-vec-list c-vec-list))
    (list
     (list :d-vec-eq d-vec-eq)
     (list :f-mat-eq f-mat-eq))
    ))

(defun get-constraint-coeff-for-environment-collision
  (&key
   (debug? nil)
   )
  "return coeff in vector-pose format."
  (let* ((vertices-list
          (mapcar #'(lambda (x) (send *obj* :inverse-transform-vector x)) (send *obj* :vertices)))
         (a-vec-list
          (make-list (length vertices-list) :initial-element *ez*))
         (b-vec-list
          vertices-list)
         (c-vec-list
          (make-list (length vertices-list) :initial-element *ez*))
         (d-list
          (make-list (length vertices-list) :initial-element 0))
         (d-vec-ineq)
         (f-mat-ineq)
         )
    (when debug?
      (pprint
       (list
        (list :a-vec-list a-vec-list)
        (list :b-vec-list b-vec-list)
        (list :c-vec-list c-vec-list)
        (list :d-list d-list)
        )))
    (setq d-vec-ineq (coerce d-list float-vector))
    (setq f-mat-ineq (convert-constraint-coeff-list-from-se3-to-vector a-vec-list b-vec-list c-vec-list))
    (list
     (list :d-vec-ineq d-vec-ineq)
     (list :f-mat-ineq f-mat-ineq))
    ))

(defun merge-constraint-coeff
  (constraint-coeff-list)
  (let* ((f-mat-eq-ret)
         (d-vec-eq-ret)
         (f-mat-ineq-ret)
         (d-vec-ineq-ret)
         )
    (dolist (constraint-coeff constraint-coeff-list)
      (let* ((f-mat-eq (cadr (assoc :f-mat-eq constraint-coeff)))
             (d-vec-eq (cadr (assoc :d-vec-eq constraint-coeff)))
             (f-mat-ineq (cadr (assoc :f-mat-ineq constraint-coeff)))
             (d-vec-ineq (cadr (assoc :d-vec-ineq constraint-coeff)))
             )
        (when f-mat-eq
          (cond (f-mat-eq-ret
                 (setq f-mat-eq-ret (concatenate-matrix-column f-mat-eq-ret f-mat-eq))
                 )
                (t
                 (setq f-mat-eq-ret f-mat-eq)
                 )))
        (when f-mat-eq
          (cond (d-vec-eq-ret
                 (setq d-vec-eq-ret (concatenate float-vector d-vec-eq-ret d-vec-eq))
                 )
                (t
                 (setq d-vec-eq-ret d-vec-eq)
                 )))
        (when f-mat-ineq
          (cond (f-mat-ineq-ret
                 (setq f-mat-ineq-ret (concatenate-matrix-column f-mat-ineq-ret f-mat-ineq))
                 )
                (t
                 (setq f-mat-ineq-ret f-mat-ineq)
                 )))
        (when f-mat-ineq
          (cond (d-vec-ineq-ret
                 (setq d-vec-ineq-ret (concatenate float-vector d-vec-ineq-ret d-vec-ineq))
                 )
                (t
                 (setq d-vec-ineq-ret d-vec-ineq)
                 )))
        ))
    (list
     (list :d-vec-eq d-vec-eq-ret)
     (list :f-mat-eq f-mat-eq-ret)
     (list :d-vec-ineq d-vec-ineq-ret)
     (list :f-mat-ineq f-mat-ineq-ret))
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

(defun convert-constraint-coeff-list-from-se3-to-vector
  (a-vec-list b-vec-list c-vec-list)
  (let* ((f-mat-eq)
         )
    (dotimes (i (length a-vec-list))
      (cond (f-mat-eq
             (setq f-mat-eq (concatenate-matrix-column f-mat-eq (convert-constraint-coeff-from-se3-to-vector
                                                                 (elt a-vec-list i) (elt b-vec-list i) (elt c-vec-list i))))
             )
            (t
             (setq f-mat-eq (convert-constraint-coeff-from-se3-to-vector
                             (elt a-vec-list i) (elt b-vec-list i) (elt c-vec-list i)))
             ))
      )
    f-mat-eq
    ))


;; sensor ;;;;;;;;;;;;;;;;;;
(defun get-random-vision-info
  (&key (edge-num 3)
        (face-num 0))
  (let* ((edge-list)
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

(defun get-grasp-info
  ()
  (send (send *obj* :get :grasp-point) :worldpos)
  )


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
(defun get-pose-vec-from-coords
  (&optional (c (send (send *obj* :worldcoords) :copy-worldcoords)))
  (let* ((pos (send c :worldpos))
         (rot (send c :worldrot))
         )
    (concatenate float-vector pos (mat2vec rot))
    ))

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


;; draw ;;;;;;;;;;;;;;;;;;
(defun draw-vision-info
  (vision-info
   &key
   (draw-estimation-result? nil)
   )
  (let* ((edge-list
          (elt vision-info 0))
         (face-list
          (elt vision-info 1))
         )
    (dolist (edge0 edge-list)
      (let* ((edge-param (elt edge0 0))
             (edge-id (elt edge0 1))
             )
        (send (elt (send *obj* :edges) edge-id) :draw-on :flush t :color #f(1 0 0) :width 12.0)
        (when draw-estimation-result?
          (send (elt (send *model-obj* :edges) edge-id) :draw-on :flush t :color #f(0 1 0) :width 12.0))
        ))
    (dolist (face0 face-list)
      (let* ((face-param (elt face0 0))
             (face-id (elt face0 1))
             )
        (send (elt (send *obj* :faces) face-id) :draw-on :flush t :color #f(0.6 0.2 0.2) :width 4.0)
        (when draw-estimation-result?
          (send (elt (send *model-obj* :faces) face-id) :draw-on :flush t :color #f(0.2 0.6 0.2) :width 4.0))
        ))
    ))

(defun draw-grasp-info
  (&key
   (draw-estimation-result? nil)
   )
  (send (send (send *obj* :get :grasp-point) :worldpos) :draw-on :flush t :color #f(0 0 1) :width 10 :size 20)
  (when draw-estimation-result?
    (send (send (send *model-obj* :get :grasp-point) :worldpos) :draw-on :flush t :color #f(0.2 0.2 0.6) :width 10 :size 20)
    )
  )

;; test ;;;;;;;;;;;;;;;;;;
(defun test
  (&key (edge-num 0)
        (face-num 1)
        (use-grasp? nil)
        (use-environment-collision? nil)
        )
  (setup-object)
  (let* ((vision-info
          (get-random-vision-info :edge-num edge-num :face-num face-num))
         (constraint-coeff-vision
          (get-constraint-coeff-from-vision-info vision-info :debug? nil))
         (grasp-info
          (get-grasp-info))
         (constraint-coeff-grasp
          (get-constraint-coeff-from-grasp-info grasp-info :debug? nil))
         (constraint-coeff-environment-collision
          (get-constraint-coeff-for-environment-collision))
         (constraint-coeff
          (merge-constraint-coeff
           (append nil
                   (list constraint-coeff-vision)
                   (if use-grasp? (list constraint-coeff-grasp))
                   (if use-environment-collision? (list constraint-coeff-environment-collision))
                   ))
          )
         (pose-vec
          (calc-pose-vec-with-nlopt constraint-coeff :debug? nil))
         ;; (pose-vec
         ;;  (calc-pose-vec-with-qp constraint-coeff :debug? nil))
         (est-pose
          (refine-pose-for-orthogonal pose-vec constraint-coeff))
         (est-pos (cadr (assoc :t-vec est-pose)))
         (est-rot (cadr (assoc :r-mat est-pose)))
         (est-rpy (coerce (car (rpy-angle est-rot)) float-vector))
         (est-pose-vec (concatenate float-vector est-pos (mat2vec est-rot)))
         (gt-pos (send *obj* :worldpos))
         (gt-rpy (coerce (car (rpy-angle (send *obj* :worldrot))) float-vector))
         )

    ;; apply result
    (send *model-obj* :newcoords (make-coords :pos est-pos :rot est-rot))
    (send *irtviewer* :draw-objects)

    ;; draw result
    (draw-vision-info vision-info :draw-estimation-result? t)
    (when use-grasp?
      (draw-grasp-info :draw-estimation-result? t))

    ;; print result
    (warning-message 2 "estimation result:~%")
    (warning-message 9 "pos~%")
    (format-array est-pos "estimated:    ")
    (format-array gt-pos "ground truth: ")
    (warning-message 9 "rpy~%")
    (format-array est-rpy "estimated:    ")
    (format-array gt-rpy "ground truth: ")

    ;; check result
    (let* ((f-mat-eq (cadr (assoc :f-mat-eq constraint-coeff)))
           (d-vec-eq (cadr (assoc :d-vec-eq constraint-coeff)))
           (f-mat-ineq (cadr (assoc :f-mat-ineq constraint-coeff)))
           (d-vec-ineq (cadr (assoc :d-vec-ineq constraint-coeff)))
           (refined-pose-vec est-pose-vec)
           (ret-without-refine)
           (ret-with-refine)
           (ret-ground-truth)
           )
      (when f-mat-eq
        (setq ret-without-refine (v+ (transform f-mat-eq pose-vec) d-vec-eq))
        (setq ret-with-refine (v+ (transform f-mat-eq refined-pose-vec) d-vec-eq))
        (setq ret-ground-truth (v+ (transform f-mat-eq (get-pose-vec-from-coords)) d-vec-eq))
        (warning-message 2 "equality constraint:~%")
        (format-array ret-without-refine (format nil "constraint error without refine [~a]: " (norm ret-without-refine)))
        (format-array ret-with-refine (format nil "constraint error with refine [~a]: " (norm ret-with-refine)))
        (format-array ret-ground-truth (format nil "ground truth [~a]: " (norm ret-ground-truth)))
        )

      (when f-mat-ineq
        (setq ret-without-refine (v+ (transform f-mat-ineq pose-vec) d-vec-ineq))
        (setq ret-with-refine (v+ (transform f-mat-ineq refined-pose-vec) d-vec-ineq))
        (setq ret-ground-truth (v+ (transform f-mat-ineq (get-pose-vec-from-coords)) d-vec-ineq))
        (warning-message 2 "inequality constraint:~%")
        (format-array ret-without-refine (format nil "constraint error without refine [~a]: " (apply #'min (coerce ret-without-refine cons))))
        (format-array ret-with-refine (format nil "constraint error with refine [~a]: " (apply #'min (coerce ret-with-refine cons))))
        (format-array ret-ground-truth (format nil "ground truth [~a]: " (apply #'min (coerce ret-ground-truth cons))))
        )

      (warning-message 2 "constraint matrix:~%")
      (when f-mat-eq (format-array f-mat-eq "f-mat-eq: "))
      (when d-vec-eq (format-array d-vec-eq "d-vec-eq: "))
      (when f-mat-ineq (format-array f-mat-ineq "f-mat-ineq: "))
      (when d-vec-ineq (format-array d-vec-ineq "d-vec-ineq: "))
      )
    ))



#|
(format-array (cadr (assoc :f-mat-eq (get-constraint-coeff-from-vision-info (get-random-vision-info)))))
(get-pose-in-vector-format-with-qp (get-constraint-coeff-from-vision-info (get-random-vision-info)) :debug? t)
(refine-pose-for-orthogonal (get-pose-in-vector-format-with-qp (get-constraint-coeff-from-vision-info (get-random-vision-info)) :debug? nil))
