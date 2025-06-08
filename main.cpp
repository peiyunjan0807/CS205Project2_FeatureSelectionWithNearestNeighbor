//
//  main.cpp
//  CS205_project2
//
//  Created by Pei-Yun Jan on 2025/5/11.
//

#include <iostream>
#include <iomanip>
#include "algorithms.hpp"
using namespace std;

int main(int argc, const char * argv[]) {
    // insert code here...

    cout<<"Welcome to Bertie Woosters Feature Selection Algorithm.\n";
    cout<<"Type in the name of the file to test : ";

    string fileName;
    getline(cin, fileName);

    cout<<"\nType the number of the algorithm you want to run.\n";
    cout<<"    1)  Forward Selection\n";
    cout<<"    2)  Backward Elimination\n\n";
    string choice;
    getline(cin, choice);

    vector<vector<double>> x;
    vector<double> y;
    if(!loadData(fileName, x, y)){
        cerr<<"Error: cannot find file "<<fileName<<"\n";
        return 1;
    }

    int nFeatures  = static_cast<int>(x[0].size());
    int nInstances = static_cast<int>(x.size());
    cout<<"\nThis dataset has "<<nFeatures<<" features (not including the class attribute), with "<<nInstances<<" instances.\n";

    zNormalize(x);
    double initialAcc = nnLeaveOneOutCV(x, y)*100.0;
    cout<<"\nRunning nearest neighbor with all "<<nFeatures<<" features, using \"leaving-one-out\" ""evaluation, I get an accuracy of "<<fixed<<setprecision(1)<<initialAcc<<"%\n\n";

    if(choice=="1"){
        forwardSelection(x, y);
    }
    else if(choice=="2"){
        backwardElimination(x, y);
    }
    else{
        cerr << "Invalid choice.\n";
    }
    
    return 0;
}

