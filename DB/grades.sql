DROP TABLE Rawscores;
CREATE TABLE Rawscores(
        SSN	VARCHAR(4) primary key,
        LName   VARCHAR(11),
        FName   VARCHAR(11),
        Section VARCHAR(3),
	HW1     DECIMAL(5,2),
	HW2a    DECIMAL(5,2),
	HW2b 	DECIMAL(5,2),
	Midterm DECIMAL(5,2),
	HW3	DECIMAL(5,2),
	FExam	DECIMAL(5,2)
        );

INSERT INTO Rawscores VALUES ('9176', 'Epp', 'Eric', '415', 99, 79, 31, 99, 119, 199);
INSERT INTO Rawscores VALUES ('5992', 'Lin', 'Linda', '415', 98, 71, 29, 83, 105, 171);
INSERT INTO Rawscores VALUES ('3774', 'Adams', 'Abigail', '315', 85, 63, 27, 88, 112, 180);
INSERT INTO Rawscores VALUES ('1212', 'Osborne', 'Danny', '315', 29, 31, 12, 66, 61, 106);
INSERT INTO Rawscores VALUES ('4198', 'Wilson', 'Amanda', '315', 84, 73, 27, 87, 115, 172);
INSERT INTO Rawscores VALUES ('1006', 'Nielsen', 'Bridget', '415', 93, 76, 28, 95, 111, 184);
INSERT INTO Rawscores VALUES ('8211', 'Clinton', 'Chelsea', '415', 100, 80, 32, 100, 120, 200);
INSERT INTO Rawscores VALUES ('1180', 'Quayle', 'Jonathan', '315', 50, 40, 16, 55, 68, 181);
INSERT INTO Rawscores VALUES ('0001', 'TOTAL', 'POINTS', '415', 100, 80, 32, 100, 120, 200);
INSERT INTO Rawscores VALUES ('0002', 'WEIGHT','OFSCORE','415', 0.1,0.1,0.05,0.25,0.1,0.4);

drop table Passwords;
create table Passwords (
  CurPasswords  VARCHAR(15)
);

INSERT INTO Passwords VALUES ('OpenSesame');
INSERT INTO Passwords VALUES ('GuessMe');
INSERT INTO Passwords VALUES ('ImTheTA');
