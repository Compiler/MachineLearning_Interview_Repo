#include <string.h>
#include <stdio.h>
void bubbleSort(char *arr[], int n) 
{
   int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 1; j < n - i - 1; j++)
        {
            char temp[100] ="";
            if (strcmp(arr[j],arr[j+1]) > 0)
                {
                strcpy(temp, arr[j]);
                strcpy(arr[j], arr[j+1]);
                strcpy(arr[j+1], temp);
                }
        }
    }
}

int main(int argc, char *argv[])
{
    bubbleSort(argv,argc);
    for (int i = 1; i < argc;i++){
        printf("%s\n",argv[i]);
    }
    return 0;
}