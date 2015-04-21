// Gaga: "grosse ambiance" genetic algorithm library
// Copyright (c) Jean Disset 2015, All rights reserved.

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library.

#ifndef TOOLS_H
#define TOOLS_H

#include <iostream>
#include <iomanip>
#include <random>
#include <sys/types.h>
#include <unistd.h>
#include <assert.h>
#include <vector>

#define PURPLE "\033[1;35m"
#define BLUE "\033[34m"
#define GREY "\033[1;30m"
#define YELLOW "\033[1;33m"
#define RED "\033[1;31m"
#define CYAN "\033[36m"
#define CYANBOLD "\033[1;36m"
#define GREEN "\033[32m"
#define GREENBOLD "\033[1;32m"
#define NORMAL "\033[0m"
#define CUR_HOME "\x1b[H"
#define CUR_UPN(n) "\x1b[" << n << "A"
#define CUR_DWN(n) "\x1b[" << n << "B"
#define CUR_POS(nl, nc) CUR_HOME << "\x1b[" << nl << ";" << nc << "H"
#define CUR_POSV(n) "\x1b[" << n << "d"

namespace GAGA {
typedef std::vector<std::vector<double>> fpType;          // footprints for novelty
typedef std::vector<std::pair<fpType, double>> archType;  // collection of footprints
extern std::default_random_engine globalRand;
}

#endif
