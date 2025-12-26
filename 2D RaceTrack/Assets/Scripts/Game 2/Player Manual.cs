using UnityEngine;

namespace two{
public class PlayerManual : MonoBehaviour
{
    float moveSpeed = 5f;
    float rotateSpeed = 100f;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        float vertical = Input.GetAxis("Vertical"); 
        float horizontal = Input.GetAxis("Horizontal");

        transform.Translate(Vector2.up * vertical * moveSpeed * Time.fixedDeltaTime);
        transform.Rotate(Vector2.up, -horizontal * rotateSpeed * Time.fixedDeltaTime);
    }
}}
