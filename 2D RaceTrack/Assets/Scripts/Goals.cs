using UnityEngine;

namespace one{public class Goals : MonoBehaviour
{
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    public int goalNumber;
    void Start()
    {
       // Car car = GetComponent<Car>();
    }

    // Update is called once per frame
    void Update()
    {
        if (goalNumber == Car.score)
        {
            GetComponent<SpriteRenderer>().color = Color.white;
        }
        else if (goalNumber < Car.score)
        {
            GetComponent<SpriteRenderer>().color = new Color(0.68f, 0.85f, 0.90f); // Light blue
        }
        else if (goalNumber > Car.score)
        {
            GetComponent<SpriteRenderer>().color = new Color(0.55f, 0f, 0f); // Dark red
        }
    }
}}
