(defun estimate-plane-with-eigen
  ()

  ;; plane parameter
  (setq *p* (float-vector 100 200 300))
  (setq *n* (normalize-vector (float-vector 123 321 213)))

  (let* ((plane-dir1 (normalize-vector (v* *n* (float-vector 1 0 0))))
         (plane-dir2 (normalize-vector (v* *n* plane-dir1)))
         (plane-point-list)
         (plane-point-num 1000)
         (plane-point-cube-list)
         (plane-range-norm 1000.0)
         (noize-norm 0.0)
         ;; (noize-norm 50.0)
         )
    ;; generate random points on plane
    (dotimes (i plane-point-num)
      (push
       (v+
        (v+ *p*
            (v+ (scale (- (random plane-range-norm) (/ plane-range-norm 2.0)) plane-dir1)
                (scale (- (random plane-range-norm) (/ plane-range-norm 2.0)) plane-dir2)))
        (float-vector (- (random (float noize-norm)) (/ noize-norm 2.0))
                      (- (random (float noize-norm)) (/ noize-norm 2.0))
                      (- (random (float noize-norm)) (/ noize-norm 2.0))))
       plane-point-list)

      )
    ;; (pprint plane-point-list)

    ;; visualize points on plane
    (dolist (plane-point plane-point-list)
      (let* ((plane-point-cube (make-cube 5 5 5))
             )
        (send plane-point-cube :newcoords (make-coords :pos plane-point))
        (push plane-point-cube plane-point-cube-list)
        ))
    (objects plane-point-cube-list)

    ;; estimate plane from points
    (let* (;; mean
           (point-mean
            (scale (/ 1.0 plane-point-num) (reduce #'v+ plane-point-list)))
           (point-square-mean-xx ;; mean of x^2, y^2, z^2
            (scale (/ 1.0 plane-point-num)
                   (reduce #'v+
                           (mapcar #'(lambda (point) (map float-vector #'* point point))
                                   plane-point-list))))
           (point-square-mean-xy ;; mean of xy^2, yz^2, zx^2
            (scale (/ 1.0 plane-point-num)
                   (reduce #'v+
                           (mapcar #'(lambda (point) (map float-vector #'* point (float-vector (elt point 1) (elt point 2) (elt point 0))))
                                   plane-point-list))))
           ;; variance and covariance
           (variance-xyz
            (map float-vector #'(lambda (psm pm) (- psm (* pm pm))) point-square-mean-xx point-mean))
           (variance-x (elt variance-xyz 0))
           (variance-y (elt variance-xyz 1))
           (variance-z (elt variance-xyz 2))
           (covariance-xyz
            (map float-vector #'(lambda (psm pmx pmy) (- psm (* pmx pmy))) point-square-mean-xy point-mean (float-vector (elt point-mean 1) (elt point-mean 2) (elt point-mean 0))))
           (covariance-xy (elt covariance-xyz 0))
           (covariance-yz (elt covariance-xyz 1))
           (covariance-zx (elt covariance-xyz 2))
           (variance-covariance-matrix
            (matrix (float-vector variance-x covariance-xy covariance-zx)
                    (float-vector covariance-xy variance-y covariance-yz)
                    (float-vector covariance-zx covariance-yz variance-z)))
           ;; eigen
           (eigen-ret (eigen-decompose variance-covariance-matrix))
           ;; view
           (point-mean-cube (make-cube 20 20 20))
           (plane-normal-cube (make-cube 10 10 100))
           )
      ;; (format-array variance-covariance-matrix)
      (print (car eigen-ret))
      (format-array (cadr eigen-ret))

      (send point-mean-cube :newcoords (make-coords :pos point-mean))
      (send point-mean-cube :set-color (float-vector 1 0 0))

      (send plane-normal-cube :newcoords (make-coords :pos (copy-seq point-mean) :rot (cadr eigen-ret)))
      (send plane-normal-cube :translate (float-vector 0 0 50) :local)
      (send plane-normal-cube :set-color (float-vector 1 0 0))

      (objects point-mean-cube)
      (objects plane-normal-cube)
      )

    ;; visualize truth plane
    (let* ((point-mean-cube (make-cube 20 20 20))
           (plane-normal-cube (make-cube 10 10 100))
           )
      (send point-mean-cube :newcoords (make-coords :pos *p*))
      (send point-mean-cube :set-color (float-vector 0 1 0))

      (send plane-normal-cube :newcoords (orient-coords-to-axis (make-coords :pos (copy-seq *p*)) *n* :z))
      (send plane-normal-cube :translate (float-vector 0 0 50) :local)
      (send plane-normal-cube :set-color (float-vector 0 1 0))

      (objects point-mean-cube)
      (objects plane-normal-cube)
      )

    t))
