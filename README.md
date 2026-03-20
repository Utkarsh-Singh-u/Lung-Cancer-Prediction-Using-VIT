# Database Entity-Relationship Diagram

This diagram represents the structure of the FACULTY, COURSE, and STUDENT entities and their specified relationships.

```mermaid
erDiagram
    FACULTY {
        int fid
        string name
        string dependent
        string salary
        string dept
    }
    COURSE {
        int cid
        string name
        string sem
        string year
        string Fa
    }
    STUDENT {
        int sid
        string name
        string dept
        string sem
        int year
        double gradepoint
        int grade
    }

    FACULTY ||--|| COURSE : teaches
    STUDENT ||--|| COURSE : enrolled
