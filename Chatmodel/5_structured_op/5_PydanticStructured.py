from pydantic import BaseModel, EmailStr ,Field
from typing import Optional
class stu(BaseModel):
    name:str='kashish'
    age:Optional[int]=None
    email:Optional[EmailStr] = None
    gpa:Optional[float]=Field(default=5,gt=0,lt=10,description='A decimal value representing the cgpa of the student')
   
     

new_stu={'name':'mina'}
student=stu(**new_stu)
print(student.name)


#default value
new2={}
student2=stu(**new2)
print(student2.name)

#optional
new3={'age':32}
student3=stu(**new3)
print(student3)

# Coercing
new4={'age':'32'}
stu4=stu(**new4)
print(stu4)

# EmailStr
new5={'age':32, 'email':'adb@gmail.com'}
stu5=stu(**new5)
print(stu5)

# Field
new6={'age':32, 'gpa':5}
stu6=stu(**new6)
print(stu6)

new_all={'name':'mina','age':32, 'gpa':5,'email':'adb@gmail.com'}
student1=stu(**new_all)
# contvert to dict
stu_dict=dict(student1)
print(stu_dict['age'])

studen_json = student1.model_dump_json()