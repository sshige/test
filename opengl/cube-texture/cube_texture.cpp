#include <opencv2/opencv.hpp>
#include <GL/glut.h>
#include <cstdio>

#define WIDTH 640
#define HEIGHT 640

GLfloat lightpos[] = { -1200.0, 150.0, -500.0, 1.0 };
GLuint texture[1];
int window_id;
cv::VideoCapture cap;

// キューブの頂点情報。
static const GLdouble aCubeVertex[][3] = {
  { -20.0, -20.0, -20.0 },
  { 20.0, -20.0, -20.0 },
  { 20.0, 20.0, -20.0 },
  { -20.0, 20.0, -20.0 },
  { -20.0, -20.0, 20.0 },
  { 20.0, -20.0, 20.0 },
  { 20.0, 20.0, 20.0 },
  { -20.0, 20.0, 20.0 }
};
// キューブの面。
static const int aCubeFace[][4] = {
  { 0, 1, 2, 3 },
  { 1, 5, 6, 2 },
  { 5, 4, 7, 6 },
  { 4, 0, 3, 7 },
  { 4, 5, 1, 0 },
  { 3, 2, 6, 7 }
};
// キューブに対する法線ベクトル。
static const GLdouble aCubeNormal[][3] = {
  { 0.0, 0.0,-1.0 },
  {-1.0, 0.0, 0.0 },
  { 0.0, 0.0,-1.0 },
  { 1.0, 0.0, 0.0 },
  { 0.0, 1.0, 0.0 },
  { 0.0,-1.0, 0.0 }
};

static const GLdouble aTextureVertex[][2] = {
  {0.0, 1.0},
  {0.0, 0.0},
  {1.0, 0.0},
  {1.0, 1.0}
};

void displayFunc(void)
{
  {
    cv::Mat frame;

    // capture camera image
    cap >> frame; // get a new frame from camera
    cv::flip(frame, frame, 0); // for opengl
    cv::cvtColor(frame, frame, CV_BGR2RGB); // for opengl

    // generate texture
    glGenTextures(1, &texture[0]);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    gluBuild2DMipmaps(GL_TEXTURE_2D, 3, frame.cols, frame.rows, GL_RGB, GL_UNSIGNED_BYTE, frame.data);
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, WIDTH, HEIGHT);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  //視野角,アスペクト比(ウィンドウの幅/高さ),描画する範囲(最も近い距離,最も遠い距離)
  gluPerspective(30.0, (double)WIDTH / (double)HEIGHT, 1.0, 1000.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  //視点の設定
  gluLookAt(150.0,100.0,-200.0, //カメラの座標
            0.0,0.0,0.0, // 注視点の座標
            0.0,1.0,0.0); // 画面の上方向を指すベクトル

  //ライトの設定
  glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

  // キューブの頂点を描画
  int texture_face_id = 0;
  glEnable(GL_TEXTURE_2D);
  glBegin( GL_QUADS );
  for (size_t i = 0; i < 2; ++i) {
    glNormal3dv( aCubeNormal[i] );// 法線ベクトルをキューブに当てる。
    for (size_t j = 0; j < 4; ++j) {
      if (i == texture_face_id) {
        glTexCoord2dv( aTextureVertex[j] );
      }
      glVertex3dv( aCubeVertex[ aCubeFace[i][j] ] );
    }
  }
  glEnd();
  glDisable(GL_TEXTURE_2D);

  glutSwapBuffers();
}

void keyFunc(unsigned char key , int x , int y)
{
  if (key == 'q') {
    cv::destroyAllWindows();
    glutDestroyWindow(window_id);
    exit(0);
  }
}

void idleFunc(void)
{
  glutPostRedisplay();
}

void Init(){
  glClearColor(0.3f, 0.3f, 0.3f, 1.0f);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
}

int main(int argc, char *argv[])
{
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  window_id = glutCreateWindow("cube_texture window");

  cap.open(0); //デバイスのオープン
  if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
  {
    //読み込みに失敗したときの処理
    printf("failed to open camera.\n");
    return -1;
  }

  glutDisplayFunc(displayFunc);
  glutKeyboardFunc(keyFunc);
  glutIdleFunc(idleFunc);
  Init();
  glutMainLoop();
  return 0;
}
