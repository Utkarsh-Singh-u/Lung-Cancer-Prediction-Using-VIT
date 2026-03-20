# ER Diagram (Diamond Style)

```mermaid
flowchart LR

    EMP[EMP]
    COMP[COMP]

    WORKS_IN{works in}

    EMP --> WORKS_IN
    WORKS_IN --> COMP

    EMP --- e_id((e_id))
    EMP --- e_name((e_name))
    EMP --- salary((salary))

    COMP --- c_id((c_id))
    COMP --- c_name((c_name))
```

# Database Entity-Relationship Diagram

This diagram represents the structure of the FACULTY, COURSE, and STUDENT entities and their specified relationships.

```mermaid
erDiagram
    EMPLOYEE{
        int id
        varchar(50) name
        int phno
        int salary
    }
    
    FACULTY {
        int fid
        varchar(50) name
        varchar(50) dependent
        int salary
        varchar(50) dept
    }
    COURSE {
        int cid
        varchar(50) name
        varchar(50) sem
        int year
        int FacultyId
    }
    STUDENT {
        int sid
        varchar(50) name
        varchar(50) dept
        varchar(50) sem
        int year
        double gradepoint
        int grade
    }

    FACULTY ||--|| COURSE : teaches
    STUDENT ||--|| COURSE : enrolled
```

# ER Diagram - University System

This ER diagram represents the relationship between **Students**, **Faculty**, and **Department**.

## 📊 ER Diagram

```mermaid
erDiagram

    STUDENTS {
        int id
        varchar(50) name
        varchar(50) sems
        int emrolled_year
        double creadit
        int dept_no
        int facul_id
    }

    FACULTY {
        int f_id
        string f_name
        string research
    }

    DEPARTMENT {
        int d_no
        string d_name
        int no_of_students
    }

    STUDENTS }o--|| FACULTY : "advised by"
    STUDENTS }o--|| DEPARTMENT : "belongs to"
```


# ER Diagram - Car & Customer System

This ER diagram represents the relationship between **Car** and **Customer**.

## 📊 ER Diagram

```mermaid
erDiagram

    CAR {
        int car_id
        string brand
        int cost
        int ins_amout
    }

    CUSTOMER {
        int c_id
        string c_name
    }

    CUSTOMER ||--o{ CAR : "owns"
```

# ER Diagram - EMP & COMP System

This ER diagram represents the relationship between **EMP (Employee)** and **COMP (Company)**.

## 📊 ER Diagram

```mermaid
erDiagram

    EMP {
        int e_id
        string e_name
        string salary
        string e_city
    }

    COMP {
        int c_id
        string c_name
        string c_city
    }

    EMP }o--|| COMP : "works in"
```
