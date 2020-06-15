# Docs

## Overview 
*docs* folder is being used for generating Doxygen comments into the *HTML* webpage. This folder holds mainly the information about Doxygen and its settings.

## Commenting code

http://www.doxygen.nl/manual/commands.html

### Commenting style
All comments for Doxygen have to use three slash symbols (`///`).

### Arguments
Doxygen arguments used in the Decision Maker project:
* `\namespace <name>` indicates that a comment block contains documentation for a namespace with name `<name>`.
* `\class <name>` indicates that a comment block contains documentation for a class with name `<name>`.
* `\struct <name>` indicates that a comment block contains documentation for a struct with name `<name>`.
* `\enum <name>` indicates that a comment block contains documentation for an enumeration with name `<name>`.
* `\file <name>` indicates that a comment block contains documentation for a source or header file with name `<name>`.
* `\brief { brief description }` starts a paragraph that serves as a brief description.
* `\details { detailed description }` starts a paragraph that serves as a detail description.
* `\param[direction] <parameter-name> { parameter description }` command has an optional attribute, (direction), specifying the direction of the parameter. Possible values are "[in]", "[in,out]", and [out]". In our project you should specify directions in a method when the same method has multiple parameters with different directions.
* `\tparam <template-parameter-name> { description }` starts a template parameter for a class or function template parameter with name `<template-parameter-name>`, followed by a description of the template parameter.
* `\return { description of the return value }` starts a return value description for a function.
* `\responsible { list of responsible people }` starts a paragraph where one or more responsible people's names may be entered. 
* `\module <name>` indicates module name `<name>`.
* `\author { list of authors }` starts a paragraph where one or more author names may be entered.
* `\project <name>` indicates project name `<name>`.
* `\copyright { copyright description }` starts a paragraph where the copyright of an entity can be described.

More information can be found [here](http://www.doxygen.nl/manual/commands.html).

### Examples

Table of contents:
* [Header file](####Header%20file)
* [Source file](####Source%20file)
* [Template class](####Template%20class)
* [Template function](####Template%20function)

#### Header file

```cpp
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file                Student.h
/// \brief               A header file of the Student class.
/// \details             This file contains a declaration of the class which holds information about a student.
/// \responsible         Andreas Gehlsen (andreas.gehlsen@audi.de)
/// \module              DecisionMaker
/// \author              
/// \project             Automated Parking Software Development
/// \copyright           (c) 2019 Volkswagen AG (EFFP/1). All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DECMAKER_STUDENT_H
#define DECMAKER_STUDENT_H

#include <string>
#include <ctime>

/// \class Student
/// \brief This class which holds information about a student.
class Student
{
public:
    /// \brief Constructor.
    /// \param id Student's ID.
    /// \param firstName Student's first name.
    /// \param familyName Student's family name.
    /// \param birthday Student's birthday.
    Student(
        int id,
        std::string firstName,
        std::string familyName,
        time_t birthday
    );

    /// \brief Constructor for a student who has the second name.
    /// \param id Student's ID.
    /// \param firstName Student's first name.
    /// \param secondName Student's second name.
    /// \param familyName Student's family name.
    /// \param birthday Student's birthday.
    Student(
        int id,
        std::string firstName,
        std::string secondName,
        std::string familyName,
        time_t birthday
    );

    /// \brief Get the student's ID.
    /// \return Student's ID.
    int getId();

    /// \brief Get the student's first name.
    /// \return Student's first name.
    std::string getFirstName();

    /// \brief Get the student's second name.
    /// \return Student's second name.
    std::string getSecondName();

    /// \brief Get the student's family name.
    /// \return Student's family name.
    std::string getFamilyName();

    /// \brief Get the student's birthday.
    /// \return Student's birthday.
    time_t getBirthday();

    /// \brief Check if the student born before a given date.
    /// \param date Date.
    /// \return True if the student was born before the given date, otherwise - false.
    bool wasBornBefore(const time_t& date);

private:
    int _id; /// Student's ID.

    std::string _firstName;  /// Student's first name.
    std::string _secondName; /// Student's second name.
    std::string _familyName; /// Student's family name.

    time_t _birthday; /// Student's birthday.
};

#endif // DECMAKER_STUDENT_H
```

#### Source file

```cpp
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file                Student.cpp
/// \brief               A source file of the Student class.
/// \details             This file contains an implementation of the class which holds information about a student.
/// \responsible         Andreas Gehlsen (andreas.gehlsen@audi.de)
/// \module              DecisionMaker
/// \author              
/// \project             Automated Parking Software Development
/// \copyright           (c) 2019 Volkswagen AG (EFFP/1). All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Student.h"

Student::Student(
    int id,
    std::string firstName,
    std::string familyName,
    time_t birthday
) :
    _id(id),
    _firstName(firstName),
    _familyName(familyName),
    _birthday(birthday)
{
}

Student::Student(
    int id,
    std::string firstName,
    std::string secondName,
    std::string familyName,
    time_t birthday
) :
    _id(id),
    _firstName(firstName),
    _secondName(secondName),
    _familyName(familyName),
    _birthday(birthday)
{
}

int Student::getId()
{
    return _id;
}

std::string Student::getFirstName()
{
    return _firstName;
}

std::string Student::getSecondName()
{
    return _secondName;
}

std::string Student::getFamilyName()
{
    return _familyName;
}

time_t Student::getBirthday()
{
    return _birthday;
}

bool Student::wasBornBefore(const time_t& date)
{
    return _birthday < date;
}
```

#### Template class

```cpp
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file                Steve.h
/// \brief               A header file with implementation of the Steve class.
/// \details             This file contains a template class for Steve.
/// \responsible         Andreas Gehlsen (andreas.gehlsen@audi.de)
/// \module              DecisionMaker
/// \author              
/// \project             Automated Parking Software Development
/// \copyright           (c) 2019 Volkswagen AG (EFFP/1). All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DECMAKER_STEVE_H
#define DECMAKER_STEVE_H

/// \class Steve
/// \brief This class is only about Steve.
/// \tparam T Type to work with.
template<typename T>
class Steve
{
public:
    /// \brief Constructor.
    /// \param favouriteObject Steve's favourite object.
    Steve(T favouriteObject) : _favouriteObject(favouriteObject)
    {
    }

    /// \brief Get Steve's favourite object.
    /// \return Steve's favourite object.
    T getFavouriteObject()
    {
        return _favouriteObject;
    }

private:
    T _favouriteObject; /// Favourite object.
};

#endif // DECMAKER_STEVE_H
```

#### Template function

```cpp
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \file                Helper.h
/// \brief               A header file with implementation of the Helper class.
/// \details             This file contains a class with helper functions.
/// \responsible         Andreas Gehlsen (andreas.gehlsen@audi.de)
/// \module              DecisionMaker
/// \author              
/// \project             Automated Parking Software Development
/// \copyright           (c) 2019 Volkswagen AG (EFFP/1). All Rights Reserved.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DECMAKER_HELPER_H
#define DECMAKER_HELPER_H

/// \class Helper
/// \brief This class containts helper functions.
class Helper
{
public:
    /// \brief Compare two objects if they are equal.
    /// \tparam T Type to work with.
    /// \param firstObject First object.
    /// \param secondObject Second object.
    /// \return True if objects are equal, otherwise - False.
    template<typename T>
    static bool isEqual(T firstObject, T secondObject)
    {
        return firstObject == secondObject;
    }
};

#endif // DECMAKER_HELPER_H
```
