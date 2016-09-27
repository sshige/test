(defun setup
  ()
  (setq *dir-path* (format nil "~a/test/eus/dataset_cube_pose/" (unix:getenv "HOME")))

  (setq *obj* (make-cube 1000.0 1000.0 1000.0))

  (setq *image-id* 0)

  (dotimes (i (length (send *obj* :faces)))
    (send *obj* :paste-texture-to-face (elt (send *obj* :faces) i)
          :file (format nil "~a/textures/face~ac.png" *dir-path* (+ i 1)))
    )
  (objects (list *obj*))
  )

(defun put-with-random-pose
  ()
  (let* ((axis-x (normalize-vector (random-vector 1.0)))
         (axis-tmp (normalize-vector (random-vector 1.0)))
         (axis-y (normalize-vector (v* axis-x axis-tmp)))
         (axis-z (normalize-vector (v* axis-x axis-y)))
         (rot (transpose (matrix axis-x axis-y axis-z)))
         (c (make-coords :rot rot))
         )
    (send *obj* :newcoords c)
    (send *irtviewer* :draw-objects)
    ))

(defun save-data
  ()
  (send *irtviewer* :viewer :viewing :look #f(3000 3000 3000) #f(0 0 0) #f(0 0 1))
  (send *irtviewer* :draw-objects)

  (let* ((data-dir (format nil "~a/data" *dir-path*))
         (image-dir (format nil "~a/images" data-dir))
         (image-filename (format nil "~a/image~a.png" image-dir *image-id*))
         (datalist-filename (format nil "~a/data-list.txt" data-dir))
         (rpy-list (reverse (car (rpy-angle (send *obj* :worldrot)))))
         (rpy-string (format nil "[~a, ~a, ~a]" (elt rpy-list 0) (elt rpy-list 1) (elt rpy-list 2)))
         )
    (unless (lisp::directory-p data-dir)
      (unix::system (format nil "mkdir -p ~a" data-dir)))
    (unless (lisp::directory-p image-dir)
      (unix::system (format nil "mkdir -p ~a" image-dir)))
    (send *irtviewer* :viewer :viewsurface :write-to-image-file image-filename)

    (when (= *image-id* 0)
      (unix::system (format nil "rm -rf ~a" datalist-filename)))
    (with-open-file
     (f datalist-filename :direction :output :if-exists :append :if-does-not-exist :create)
     (format f "~a : ~a~%" image-filename rpy-string)
     )
    )

  (incf *image-id*)
  )

(defun test
  (&optional (true-rpy (list 0 0 0)) (ret-rpy (list 0 0 0)))
  (send *obj* :newcoords (make-coords :rpy (reverse true-rpy)))
  (send *irtviewer* :viewer :viewing :look #f(3000 3000 3000) #f(0 0 0) #f(0 0 1))
  (send *irtviewer* :draw-objects)
  (read-line)
  (send *obj* :newcoords (make-coords :rpy (reverse ret-rpy)))
  (send *irtviewer* :viewer :viewing :look #f(3000 3000 3000) #f(0 0 0) #f(0 0 1))
  (send *irtviewer* :draw-objects)
  )

(setup)

;; (dotimes (i 10000)
;;   (put-with-random-pose)
;;   (save-data)
;;   )
