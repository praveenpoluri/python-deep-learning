class Employee():
    empCount = 0 #initialization of the data member
    empSal = [];
    # Create a constructor to initialize name, family, salary, department
    def __init__(self,name,family,salary,department): # default constructor
        self.empname = name
        self.empfamily = family
        self.empsalary = salary
        self.empdepartment = department
        Employee.empCount +=1  # counts the number of employees
        Employee.empSal.append(salary)  # appends salaray attribute

    def avg_salary(self): # Create a function to average salary
        print('the average salary is')
        sumSal = 0;
        for sal in Employee.empSal:
            sumSal = sumSal+ int(sal);
        return sumSal/len(Employee.empSal)
# inherits characteristics from parent class
class FulltimeEmployee(Employee):
    def __init__(self,name,family,salary,department):
        Employee.__init__(self,name,family,salary,department)



emp1 = FulltimeEmployee('Praveen',' Poluri','7000','CS');
emp2 = FulltimeEmployee('Praneeth','Thota','5000','CS');


emp3 = Employee('Praveen',' Poluri','7000','CS');
emp4 = Employee('Praneeth','Thota','5000','CS');
print(FulltimeEmployee.empCount)
print(FulltimeEmployee.empSal)
avgSal = FulltimeEmployee.avg_salary(Employee);
print(avgSal)