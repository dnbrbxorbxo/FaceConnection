from peewee import *

# SQLite 데이터베이스 연결
db = SqliteDatabase('MathBank.db')

class BaseModel(Model):
    class Meta:
        database = db

class User(BaseModel):
    name = CharField()
    grade = IntegerField()
    parent_contact = CharField()
    user_class = CharField()
    school = CharField()
    username = CharField(unique=True)
    password = CharField()
    approved = BooleanField(default=True)  # 추가적인 필드

class Problem(BaseModel):
    title = CharField()
    description = TextField()
    difficulty = IntegerField()
    solution = TextField()

# 데이터베이스 초기화
db.connect()

#db.drop_tables([User, Problem], safe=True)  # 기존 테이블을 삭제합니다.
#db.create_tables([User, Problem], safe=True)  # 새 테이블을 생성합니다.