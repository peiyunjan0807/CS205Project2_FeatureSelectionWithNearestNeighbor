#include <bits/stdc++.h>
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>
using namespace std;

bool loadData(const string& filename, vector<vector<double>>& x, vector<double>& y){
    
    ifstream fin(filename);
    if (!fin) return false;

    string line;
    while(getline(fin, line)){
        istringstream iss(line);
        double label;
        iss>>label;
        y.push_back(label);

        vector<double> row;
        double v;
        while(iss>>v) row.push_back(v);
        x.push_back(std::move(row));
    }
    return true;
}

void zNormalize(vector<vector<double>>& data){
    
    double sum=0.0, sq=0.0;
    size_t cnt=0;
    for (auto& row : data){
        for (double v : row){
            sum += v;
            sq += v*v;
            cnt++;
        }
    }

    double mean=sum/cnt;
    double std=sqrt((sq/cnt)-(mean*mean));
    for (auto& row : data){
        for (double& v : row){
            v=(v-mean)/std;
        }
    }
}

double euclidean(const vector<double>& a, const vector<double>& b){
    double distSq=0.0;
    for(size_t i=0; i<a.size(); i++){
        distSq+=((a[i]-b[i])*(a[i]-b[i]));
    }
    return sqrt(distSq);
}

double nnLeaveOneOutCV(const vector<vector<double>>& x, const vector<double>& y, bool normalize=false){
    size_t n=x.size();
    size_t correct = 0;

    for(size_t i=0; i<n; i++){
        double bestDist=numeric_limits<double>::infinity();
        double pred=-1;

        for(size_t j=0; j<n; j++){
            if(i==j) continue;
            double d=euclidean(x[i], x[j]);
            if(d<bestDist){
                bestDist=d;
                pred=y[j];
            }
        }
        if(pred==y[i]) correct++;
    }
    return static_cast<double>(correct)/n;
}


using FeatSet=vector<int>;
vector<vector<double>>

project(const vector<vector<double>>& x, const FeatSet& feats){
    
    vector<vector<double>> sub(x.size(), vector<double>(feats.size()));
    for(size_t i=0; i<x.size(); i++){
        for(size_t j=0; j<feats.size(); j++){
            sub[i][j]=x[i][feats[j]-1]; 
        }
    }
    return sub;
}

//Forward Selection
void forwardSelection(const vector<vector<double>>& x, const vector<double>& y){
    
    FeatSet remaining, current, globalBest;
    int d=static_cast<int>(x[0].size());
    for(int f=1; f<=d; f++) remaining.push_back(f);

    double globalBestAcc=-1.0;
    cout<<"Beginning search.\n\n";

    while(!remaining.empty()){
        double localBestAcc=-1.0;
        int bestFeat=-1;

        for(int f:remaining){
            FeatSet trial=current;
            trial.push_back(f);
            auto subX=project(x, trial);
            double acc=nnLeaveOneOutCV(subX, y)*100.0;

            cout<<"    Using feature(s) {";
            for(size_t i=0; i<trial.size(); i++){
                cout<<(i ? ", " : "")<<trial[i];
            }
            cout<<"} accuracy is "<<fixed<<setprecision(1)<<acc<<"%\n";

            if(acc>localBestAcc){
                localBestAcc=acc;
                bestFeat = f;
            }
        }

        FeatSet newCurrent=current;
        newCurrent.push_back(bestFeat);

        if(localBestAcc<globalBestAcc){
            cout<<"\n(WARNING, Accuracy has decreased! Continuing search in case of local maxima)";
        }


        //current = newCurrent;
        //globalBestAcc = std::max(globalBestAcc, localBestAcc);

        current = newCurrent;
        if(localBestAcc>globalBestAcc){
            globalBestAcc=localBestAcc;
            globalBest=current;
        }
        
        cout<<"\nFeature set {";
        for(size_t i=0; i<current.size(); i++){
            cout<<(i ? ", " : "")<<current[i];
        }
        cout<<"} was best, accuracy is "<<fixed<<setprecision(1)<<localBestAcc<<"%\n\n";

        remaining.erase(remove(remaining.begin(), remaining.end(), bestFeat), remaining.end());
    }
    /*
    std::cout << "Finished search!! The best feature subset is {";
    for (size_t i = 0; i < current.size(); ++i)
        std::cout << (i ? ", " : "") << current[i];
    std::cout << "}, which has accuracy of "
              << std::fixed << std::setprecision(1)
              << globalBestAcc << "%\n";
     */
    cout<<"Finished search!! The best feature subset is {";
    for(size_t i=0; i<globalBest.size(); i++){
        cout<<(i ? ", " : "")<<globalBest[i];
    }
    cout<<"}, which has accuracy of "<<fixed<<setprecision(1)<<globalBestAcc<<"%\n";

}

//Backward Elimination
void backwardElimination(const vector<vector<double>>& x, const vector<double>& y){
    using FeatSet=vector<int>;
    FeatSet current, globalBest;

    int d=static_cast<int>(x[0].size());
    for(int f=1; f<=d; f++) current.push_back(f);

    auto subXAll=project(x, current);
    double globalBestAcc=nnLeaveOneOutCV(subXAll, y)*100.0;
    globalBest=current;

    cout<<"Calculating initial accuracy = "<<fixed<<setprecision(1)<<globalBestAcc<<"%\nBeginning search.\n";

    while(current.size()>1){
        double localBestAcc=-1.0;
        int featToRemove=-1;

        for(int f:current){
            FeatSet trial=current;
            trial.erase(remove(trial.begin(), trial.end(), f), trial.end());

            auto subX=project(x, trial);
            double acc=nnLeaveOneOutCV(subX, y)*100.0;

            cout<<"    Using feature(s) {";
            for(size_t i=0; i<trial.size(); i++){
                cout<<(i ? ", " : "")<<trial[i];
            }
            cout<<"} accuracy is "<<fixed<<setprecision(1)<<acc<<"%\n";

            if(acc>localBestAcc){
                localBestAcc=acc;
                featToRemove=f;
            }
        }

        current.erase(remove(current.begin(), current.end(), featToRemove), current.end());

        if(localBestAcc>globalBestAcc){
            globalBestAcc=localBestAcc;
            globalBest=current;
        }
        else{
            cout<<"(WARNING, Accuracy has decreased! Continuing search in case of local maxia)\n";
        }

        cout<<"Feature set {";
        for(size_t i=0; i<current.size(); i++){
            cout<<(i ? ", " : "")<<current[i];
        }
        cout<<"} was best, accuracy is "<<fixed<<setprecision(1)<<localBestAcc<<"%\n\n";
    }

    cout<<"Finished search!! The best feature subset is {";
    for(size_t i=0; i<globalBest.size(); i++){
        cout<<(i ? ", " : "")<<globalBest[i];
    }
    cout<<"}, which has accuracy of "<<fixed<<setprecision(1)<<globalBestAcc<<"%\n";
}

