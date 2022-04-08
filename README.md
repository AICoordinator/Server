# Server


## Serializer 변경사항 
로그인후 유저정보 + token 같이 가게 바꿈   ㅇㅇ    



## Todo 4/6
Login 후 User정보 + Token 같이보낼 Serializer 개발 -> 자동 로그인에 이용 + 어플 사용시 이용      




## User App 구현사항  
Django 에서 제공하는 Token Authentication이용하여 로그인,로그아웃, 회원 가입 구현    
그냥 ORM으로 User하면 Django 기능
Authentication이 제공하는 메소드 쓰기위해 기본 django User Model 쓰려다 너무 필요없는거 많아서 새로만듬(주석처리함)         

#### 사용시나리오   
1.회원 가입시 유저 고유 토큰 부여함 + 리턴해줌     
2.로그인시 DB에 저장된 token 리턴해줌    
3.로그인한 유저는 Android App에서 token Memory상에 유지함     
4.Server에 요청이 있을때마다 Token가지고 요청보냄 -> 서버는 어떤 유저인지 token으로 알 수 있음. 

#### 참고 문서 
https://medium.com/geekculture/register-login-and-logout-users-in-django-rest-framework-51486390c29    
토큰   
https://han-py.tistory.com/216     
https://eunjin3786.tistory.com/253#comment21225325    
User 모델     
https://yonghyunlee.gitlab.io/python/user-extend/       
https://docs.djangoproject.com/en/dev/topics/auth/customizing/#a-full-example     
https://milooy.wordpress.com/2016/02/18/extend-django-user-model/      

#### 로컬 적용 방법
mygration 파일 init 제외 모두 삭제     
DB 파일 모두삭제     
python manage.py makemigrations    
python manage.py migraion

python manage.py createsuperuser   -> 슈퍼유저 생성 후
admin page   
